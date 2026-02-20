import torch
import torch.nn as nn
import torchaudio
import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import random
import os
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import copy
import torch.nn.functional as F

def pcen(spec, s=0.025, alpha=0.01, delta=0, r=0.05, eps=1e-6):
    """
    Apply Per-Channel Energy Normalization (PCEN) to a spectrogram.
    
    This version uses a 1D convolution to implement the moving average filter,
    which is robust to shape errors and correctly applies the filter across
    time for each frequency bin independently.

    Parameters:
    - spec (Tensor): Input spectrogram of shape [Batch, Freq, Time] or [Batch, Channels, Freq, Time].
    - s (float): Time constant for the smoother (controls the size of the moving average window).
    - alpha (float): The gain control exponent.
    - delta (float): A bias term.
    - r (float): The root compression exponent.
    - eps (float): A small value to prevent division by zero.
    
    Returns:
    - Tensor: PCEN-processed spectrogram with the same shape as the input.
    """
    try:
        if spec.ndim not in [3, 4]:
            raise ValueError(f"Input tensor must be either 3D or 4D, but got {spec.ndim}D tensor.")

        device = spec.device
        
        # --- Reshape to a standard 3D tensor for processing ---
        orig_shape = spec.shape
        if spec.ndim == 4:
            # Flatten batch and channel dimensions to get [B*C, Freq, Time]
            spec = spec.view(-1, orig_shape[-2], orig_shape[-1])

        # --- Calculate moving average filter (M) using Conv1d ---
        # This is the core of the fix. Conv1d with 'padding=same' and groups
        # correctly applies a moving average across time for each frequency bin.
        
        in_channels = spec.shape[1]  # Number of frequency bins
        time_steps = spec.shape[2]
        
        # Kernel size is a fraction of the total time steps
        kernel_size = max(1, int(s * time_steps))

        # Create a 1D convolution layer to act as a moving average filter.
        # 'groups=in_channels' ensures that the filter is applied independently
        # to each frequency bin (channel).
        ma_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding='same',  # Guarantees output size == input size
            bias=False,
            groups=in_channels
        ).to(device)

        # Set the convolution weights to be a uniform average kernel.
        ma_conv.weight.data.fill_(1.0 / kernel_size)
        ma_conv.weight.requires_grad = False # Ensure weights are not trained

        # M is the smoothed (time-averaged) spectrogram
        M = ma_conv(spec)
        
        # --- Apply the PCEN formula ---
        # (spec / (M + eps)^alpha + delta)^r - delta^r
        pcen_spec = (spec / (M + eps).pow(alpha) + delta).pow(r) - delta**r

        # Reshape back to the original 4D shape if necessary
        if len(orig_shape) == 4:
            pcen_spec = pcen_spec.view(orig_shape)

    except Exception as e:
        print(f"Error during PCEN processing: {e}")
        print("PCEN failed, returning original spectrogram.")
        # Return the original spectrogram in its original shape
        return spec.view(orig_shape)
    
    return pcen_spec

def log_scale_spectrogram(spec):
    #logarithmic scaling on the y-axis
    # Get original dimensions
    _, original_height, original_width = spec.shape
    log_scale = torch.logspace(0, 1, steps=original_height, base=10.0) - 1
    log_scale_indices = torch.clamp(log_scale * (original_height - 1) / (10 - 1), 0, original_height - 1).long()
    log_spec = spec[:, log_scale_indices, :]# Resample spectrogram on new y-axis

    return log_spec

def resample_log_mask_to_linear(log_space_mask, linear_spec_shape):
    """
    Resamples a mask from a logarithmic frequency scale to a linear frequency scale.

    Args:
        log_space_mask (torch.Tensor): The mask from the model output [H_log, W_log].
        linear_spec_shape (tuple): The shape of the target linear spectrogram [H_linear, W_linear].

    Returns:
        torch.Tensor: The resampled mask in the linear frequency space.
    """
    linear_height, linear_width = linear_spec_shape
    
    # Create a grid of target coordinates in the linear space (normalized from -1 to 1)
    y = torch.linspace(-1, 1, linear_height, device=log_space_mask.device)
    x = torch.linspace(-1, 1, linear_width, device=log_space_mask.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # The grid_x is for the time axis and remains linear.
    # We must transform the y-coordinates of our target grid from linear space to find
    # where they should sample from in the source (logarithmic space) mask.
    
    # 1. Convert target linear coordinates from [-1, 1] to [0, 1]
    y_norm = (grid_y + 1) / 2
    
    # 2. Apply the inverse of the logspace(base=10) function to find the source coordinate.
    # The forward mapping was effectively y_log = log10(x_lin * 9 + 1).
    # This is the correct inverse mapping for the coordinates.
    log_y_norm = torch.log10(y_norm * 9 + 1)
    
    # 3. Convert the resulting source coordinates from [0, 1] back to [-1, 1] for grid_sample.
    log_grid_y = log_y_norm * 2 - 1

    # 4. Combine the transformed y coordinates with the original x coordinates.
    # The grid needs to be in (x, y) order for grid_sample.
    target_grid = torch.stack((grid_x, log_grid_y), dim=-1).unsqueeze(0)

    # Resample the source mask using the generated grid.
    # Mask needs to be in shape [N, C, H_in, W_in].
    log_space_mask_unsqueezed = log_space_mask.float().unsqueeze(0).unsqueeze(0)
    
    # Perform the resampling.
    linear_mask = F.grid_sample(
        log_space_mask_unsqueezed, 
        target_grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=False
    )
    
    return linear_mask.squeeze(0).squeeze(0)

def map_frequency_to_log_scale(original_height, freq_indices):
    # Convert frequency indices to log scale
    log_freq_indices = []
    for freq_index in freq_indices:
        # Find the relative position in the original linear scale
        relative_position = freq_index / (original_height - 1 if original_height > 1 else 1)
        
        # Map to the log scale
        log_position = torch.log10(torch.tensor(relative_position * (10 - 1) + 1))
        log_index = int(torch.round(log_position * (original_height - 1) / torch.log10(torch.tensor(10.0))))
        log_freq_indices.append(log_index)
    
    return log_freq_indices

def map_frequency_to_linear_scale(original_height, freq_indices):
    # Convert frequency indices from log scale to linear scale
    linear_freq_indices = []
    for freq_index in freq_indices:
        # Find the relative position in the log scale
        relative_position = freq_index / (original_height - 1 if original_height > 1 else 1)
        
        # Map from log scale to linear scale
        linear_position = (10 ** (relative_position * torch.log10(torch.tensor(10.0))) - 1) / 9
        linear_index = int(torch.round(linear_position * (original_height - 1)))
        linear_freq_indices.append(linear_index)
    
    return linear_freq_indices

def spec_to_pil(spec, resize=None, iscomplex=False, normalise='power_to_PCEN', color_mode='HSV'):
    if iscomplex:
        spec = torch.abs(spec)

    if normalise:
        if normalise == 'power_to_dB':
            spec = 10 * torch.log10(spec + 1e-6)
        elif normalise == 'dB_to_power':
            spec = 10 ** (spec / 10)
        elif normalise == 'power_to_PCEN':
            spec = pcen(spec)
        elif normalise == 'complex_to_PCEN':
            # square complex energy for magnitude power
            spec = torch.square(spec)
            spec = pcen(spec)

    spec = np.squeeze(spec.numpy())
    spec = np.flipud(spec)  # vertical flipping for image cos
    spec = (spec - spec.min()) / (spec.max() - spec.min())  # scale to 0 - 1

    if color_mode == 'HSV':
        value = spec
        saturation = 4 * value * (1 - value)
        hue = np.linspace(0,1,spec.shape[0])[:, np.newaxis] # linearly spaced hues over frequency range
        hue = np.tile(hue, (1, spec.shape[1]))
        hsv_spec = np.stack([hue, saturation, value], axis=-1)
        rgb_spec = hsv_to_rgb(hsv_spec)
        rgb_spec = np.clip(rgb_spec, 0, 1) # should already be here but ensure nonetheless
        spec = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
    elif color_mode == 'RGB':
        # stack array 3 times to form rgb template
        spec = np.stack([spec, spec, spec], axis=-1)
        spec = Image.fromarray(np.uint8(spec * 255), 'RGB')
    else:
        spec = Image.fromarray(np.uint8(spec * 255), 'L')

    if resize:
        spec = spec.resize(resize, Image.Resampling.LANCZOS)
    
    return spec

def spec_to_audio(spec, energy_type='power', save_to=None, normalise_rms=0.05, sample_rate=48000, specconfig={
        'n_fft': 2048,
        'win_length': 2048,
        'hop_length': 1024,
    }):
    # convert back to waveform and save to wav for viewing testing
    waveform_transform = torchaudio.transforms.GriffinLim(
        n_fft=specconfig['n_fft'],
        win_length=specconfig['win_length'],
        hop_length=specconfig['hop_length'],
        power=2.0
    )
    # if energy_type=='dB':
    #     normalise = 'dB_to_power'
    # elif energy_type=='power':
    #     normalise = None
    # spec_audio = spectrogram_transformed(spec, normalise=normalise, to_torch=True)
    # if energy_type=='complex':
    waveform = waveform_transform(spec)
    rms = torch.sqrt(torch.mean(torch.square(waveform)))
    waveform = waveform*normalise_rms/rms
    if save_to:
        # check if save_to dir exists
        if not os.path.exists(os.path.dirname(save_to)):
            # save to includes the filename
            os.makedirs(os.path.dirname(save_to))
        torchaudio.save(f"{save_to}.wav", waveform, sample_rate=sample_rate)

def band_pass_spec(spec, sample_rate=48000, lowpass_hz=None, highpass_hz=None):
    if lowpass_hz:
        lowpass = int(lowpass_hz*spec.shape[1]/(sample_rate/2))
        spec[:, lowpass:, :] = 0
    if highpass_hz:
        highpass = int(highpass_hz*spec.shape[1]/(sample_rate/2))
        spec[:, :highpass, :] = 0
    return spec

def clamp_intensity_spec(spec, clamp_intensity):
    if isinstance(clamp_intensity, (int, float)):
        spec = spec.clamp(min=clamp_intensity)
    elif isinstance(clamp_intensity, (tuple, list)):
        spec = spec.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
    return spec

def scale_intensity_spec(spec, scale_intensity):
    if isinstance(scale_intensity, (int, float)):
        spec = spec * scale_intensity
    elif isinstance(scale_intensity, (tuple, list)):
        minimum = scale_intensity[0]
        maximum = scale_intensity[1]
        spec = (spec - spec.min()) / (spec.max() - spec.min()) * (maximum - minimum) + minimum
    return spec

def set_dB(spec, set_db, unit_type='energy'):
    if unit_type == 'energy':
        power = torch.mean(torch.square(spec))
        normalise_power = torch.sqrt(10 ** (set_dB / 10) / power)
    elif unit_type == 'power':
        power = torch.mean(spec)
        normalise_power = 10 ** (set_db / 10) / power
    spec = spec * normalise_power
    return spec

def complex_spectrogram_transformed(
        spec,
        lowpass=None,
        highpass=None,
        scale_intensity=None,
        clamp_intensity=None,
        set_snr_db=None,
        scale_mean=None,
        set_rms=None,
        normalise=None,
        to_torch=False,
        to_numpy=False,
        to_pil=False,
        resize=None
    ):
    # scaling and clamping is done BEFORE normalisation

    real_part = spec.real
    imaginary_part = spec.imag

    if highpass:
        hp = int(highpass*spec.shape[1]/24000)
        real_part[:, :hp, :] = 0
        imaginary_part[:, :hp, :] = 0
    if lowpass:
        lp = int(lowpass*spec.shape[1]/24000)
        real_part[:, lp:, :] = 0
        imaginary_part[:, lp:, :] = 0

    if isinstance(scale_intensity, int) or isinstance(scale_intensity, float) or (isinstance(scale_intensity, torch.Tensor) and scale_intensity.shape==[]): 
        real_part = real_part * scale_intensity
        imaginary_part = imaginary_part * scale_intensity
    elif isinstance(scale_intensity, tuple) or isinstance(scale_intensity, list):
        minimum = scale_intensity[0]
        maximum = scale_intensity[1]
        real_part = (real_part - real_part.min()) / (real_part.max() - real_part.min()) * (maximum - minimum) + minimum
        imaginary_part = (imaginary_part - imaginary_part.min()) / (imaginary_part.max() - imaginary_part.min()) * (maximum - minimum) + minimum
    if isinstance(clamp_intensity, int) or isinstance(clamp_intensity, float):
        real_part = real_part.clamp(min=clamp_intensity)
        imaginary_part = imaginary_part.clamp(min=clamp_intensity)
    elif isinstance(clamp_intensity, tuple) or isinstance(clamp_intensity, list):
        real_part = real_part.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
        imaginary_part = imaginary_part.clamp(min=clamp_intensity[0], max=clamp_intensity[1])
    if set_snr_db:
        power = torch.mean(torch.sqrt(real_part**2 + imaginary_part**2))
        # set_snr to db
        set_snr = 10 ** (set_snr_db / 10)
        normalise_power = set_snr / power
        real_part = real_part * normalise_power
        imaginary_part = imaginary_part * normalise_power
    
    if scale_mean:
        # normalise mean value for the non-zero values
        non_zero_real = real_part[real_part != 0]
        mean_power = non_zero_real.mean()
        normalise_power = scale_mean / mean_power
        real_part = real_part * normalise_power
        imaginary_part = imaginary_part * normalise_power
    if set_rms:
        component_magnitude = torch.sqrt(real_part**2 + imaginary_part**2)
        rms = torch.sqrt(torch.mean(component_magnitude**2))
        normalise_rms_factor = set_rms / rms
        real_part = real_part * normalise_rms_factor
        imaginary_part = imaginary_part * normalise_rms_factor

    # if normalise:
    #     if normalise=='power_to_dB':
    #         real_part = 10 * torch.log10(real_part + 1e-6)
    #         imaginary_part = 10 * torch.log10(imaginary_part + 1e-6)
    #     elif normalise=='dB_to_power':
    #         real_part = 10 ** (real_part / 10)
    #         imaginary_part = 10 ** (imaginary_part / 10)
    #     elif normalise=='PCEN':
    #         real_part = pcen(real_part)
    #         imaginary_part = pcen(imaginary_part)

    if to_pil:
        return spec_to_pil(real_part + 1j * imaginary_part, resize=resize, iscomplex=True, normalise='complex_to_PCEN', color_mode='HSV')

    if to_numpy:
        return np.stack([real_part.numpy(), imaginary_part.numpy()], axis=-1)
    return real_part + 1j * imaginary_part

def spectrogram_transformed(
        spec,
        lowpass_hz=None, 
        highpass_hz=None, 
        scale_intensity=None, 
        clamp_intensity=None, 
        set_db=None,
        scale_mean=None, 
        set_rms=None,
        to_torch=False, 
        to_numpy=False, 
        to_pil=False, 
        normalise='power_to_PCEN', #only if to pil
        color_mode='HSV', #only if to pil
        log_scale=False,
        resize=None
    ):
    # [spec.shape] = [batch, freq_bins, time_bins]
    # scaling and clamping is done BEFORE normalisation
    is_numpy = isinstance(spec, np.ndarray)
    if is_numpy:
        to_numpy=True
        spec = torch.from_numpy(spec)

    if len(spec.shape) == 2:
        spec = spec.unsqueeze(0)

    if spec.is_complex():
        print('complex spectrogram')
        # spec = complex_spectrogram_transformed(
        #     spec,lowpass=lowpass,highpass=highpass,scale_intensity=scale_intensity,set_snr_db=set_snr_db,clamp_intensity=clamp_intensity,scale_mean=scale_mean,set_rms=set_rms,normalise=normalise,to_torch=to_torch,to_numpy=to_numpy,to_pil=to_pil,resize=resize
        # )
    else:
        if highpass_hz:
            spec = band_pass_spec(spec, highpass_hz=highpass_hz)
        if lowpass_hz:
            spec = band_pass_spec(spec, lowpass_hz=lowpass_hz)
        if scale_intensity:
            spec = scale_intensity_spec(spec, scale_intensity)
        if clamp_intensity:
            spec = clamp_intensity_spec(spec, clamp_intensity)
        if set_db:
            spec = set_dB(spec, set_db, unit_type='power')
        if log_scale:
            spec = log_scale_spectrogram(spec)

        if scale_mean:
            print('tried tp scale mean')
            # normalise mean value for the non-zero values
            # non_zero_spec = spec[spec != 0]
            # mean_power = non_zero_spec.mean()
            # normalise_power = scale_mean / mean_power
            # spec = spec * normalise_power
        if set_rms:
            print('tried to set rms')
            # assuming unit_type=power
            # power = torch.mean(spec)
            # normalise_power = set_rms / power
            # spec = spec * normalise_power

        if to_pil:
            # spec = np.squeeze(spec.numpy())
            # spec = np.flipud(spec) # vertical flipping for image cos
            # spec = (spec - spec.min()) / (spec.max() - spec.min()) # scale to 0 - 1
            # if to_pil=='HSV':
            #     value = spec
            #     saturation = 4 * value * (1 - value)
            #     hue = np.linspace(0,1,spec.shape[0])[:, np.newaxis] # linearly spaced hues over frequency range
            #     hue = np.tile(hue, (1, spec.shape[1]))
            #     hsv_spec = np.stack([hue, saturation, value], axis=-1)
            #     rgb_spec = hsv_to_rgb(hsv_spec)
            #     spec = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
            # else: # greyscale
            #     spec = Image.fromarray(np.uint8(spec * 255), 'L')
            # if resize:
            #     spec = spec.resize(resize, Image.Resampling.LANCZOS)
            # return spec
            return spec_to_pil(spec, resize=resize, iscomplex=False, normalise=normalise,color_mode=color_mode)
    
    if to_numpy:
        spec = np.squeeze(spec.numpy())
    return spec

def load_spectrogram(
        paths, 
        unit_type='complex', 
        rms_normalised=1, 
        random_crop=None, 
        chunk_length=None, 
        max=20,
        overlap=0.5, 
        resample_rate=48000
    ):
    
    if unit_type == 'complex':
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            power=None,  # Produces complex-valued spectrogram without reducing to power or magnitude
    )
    elif unit_type=='power':
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048, 
            win_length=2048, 
            hop_length=512, 
            power=2.0,
            window_fn=torch.hamming_window
        )
    elif unit_type=='dB':
        spec_transform = lambda x: 10 * torch.log10(torch.abs(torchaudio.transforms.Spectrogram(
            n_fft=2048, 
            win_length=2048, 
            hop_length=512, 
            power=2.0,
            window_fn=torch.hamming_window
        )(x)) + 1e-6)

    specs = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        try:
            waveform, sample_rate = torchaudio.load(path)
        except:
            print(f'couldn"t load {path}')
            return None
        # print(f'loaded {os.path.basename(path)}, sample rate {sample_rate}')
        resample = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=torch.float32).to(waveform.device)
        waveform = resample(waveform)
        if waveform.shape[0]>1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) # mono

        #randomly crop to random_crop seconds
        if random_crop: 
            cropped_samples = random_crop * resample_rate
            if waveform.shape[1] > cropped_samples:
                start = random.randint(0, waveform.shape[1] - cropped_samples)
                waveform = waveform[:, start:start+cropped_samples]
            else:
                print(f"Error: {path} is shorter than random crop length {random_crop}")
                return None
        
        # rms normalise
        if rms_normalised:
            rms = torch.sqrt(torch.mean(torch.square(waveform)))
            waveform = waveform*rms_normalised/rms

        # separate into chunks of chunk_length seconds
        if chunk_length and (waveform.shape[1] > (chunk_length * resample_rate)):
            samples_per_chunk = int(chunk_length * resample_rate)
            overlap_samples = int(samples_per_chunk * overlap)
            samples_overlap_difference = samples_per_chunk - overlap_samples
            for i in range(0, waveform.shape[1], samples_overlap_difference):
                chunk = waveform[:, i:i+samples_per_chunk]
                spec = spec_transform(chunk)
                specs.append(spec)
                if max and len(specs)>=max:
                    break

        else:
            spec = spec_transform(waveform)
            specs.append(spec)
    
    if len(specs)==1:
        return specs[0]
    return specs

def crop_overlay_waveform(
        bg_shape,
        segment,
        minimum_samples_present=48000,
    ):
    # determine the segments start position, and crop it if it extends past the background
    if segment.shape[1] > bg_shape:
        minimum_start = bg_shape - segment.shape[1]
        maximum_start = 0
    else:
        minimum_start = min(0, minimum_samples_present-segment.shape[1])
        maximum_start = max(bg_shape-segment.shape[1], bg_shape-minimum_samples_present)
    start = random.randint(minimum_start, maximum_start)
    cropped_segment = segment[:, max(0,-start) : min(bg_shape-start, segment.shape[1])]
    return cropped_segment, start

def add_noise_to_waveform(waveform, noise_power, noise_type):
    if noise_type=='white':
        noise = torch.randn_like(waveform) * torch.sqrt(torch.tensor(noise_power))
    elif noise_type=='brown':
        noise = torch.cumsum(torch.randn_like(waveform), dim=-1)
        noise = noise - noise.mean()
        noise = noise / noise.std()
        noise = noise * torch.sqrt(torch.tensor(noise_power))
    elif noise_type=='pink':
        white_noise = torch.randn_like(waveform)
        fft = torch.fft.rfft(white_noise, dim=-1)
        frequencies = torch.fft.rfftfreq(waveform.shape[-1], d=1.0)
        
        # Avoid dividing by zero frequency
        fft[:, 1:] /= torch.sqrt(frequencies[1:])
        
        pink_noise = torch.fft.irfft(fft, n=waveform.shape[-1], dim=-1)
        pink_noise = pink_noise / pink_noise.std()
        noise = pink_noise * torch.sqrt(torch.tensor(noise_power))
    return waveform + noise

def transform_waveform(waveform, 
        resample=[48000,48000], 
        specconfig={
            'n_fft':2048,
            'win_length':2048, 
            'hop_length':512
        },
        random_crop_seconds=None, 
        crop_seconds=None,
        rms_normalised=None, 
        set_db=None,
        to_spec=None, 
        add_white_noise=None, 
        add_pink_noise=None, 
        add_brown_noise=None
    ):
    if waveform.shape[0]>1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # mono
    
    if  not (resample[0]==resample[1]):
        resample_transform = torchaudio.transforms.Resample(resample[0], resample[1], dtype=torch.float32).to(waveform.device)
        waveform = resample_transform(waveform)
    
    if random_crop_seconds: 
        cropped_samples = random_crop_seconds * resample[1]
        if waveform.shape[1] > cropped_samples:
            start = random.randint(0, waveform.shape[1] - cropped_samples)
            waveform = waveform[:, start:start+cropped_samples]
        else:
            print(f"Error: shorter than random crop length {random_crop_seconds}")
            return None
    elif crop_seconds:
        print(f'cropping {crop_seconds}, waveform shape {waveform.shape}')
        start_crop, end_crop = crop_seconds
        start = int(start_crop * resample[1])
        end = int(end_crop * resample[1])
        if end > waveform.shape[1]:
            print(f"Error: end crop {end_crop} is longer than waveform")
            return None
        waveform = waveform[:, start:end]
        print(f'cropped waveform shape {waveform.shape}')
    if rms_normalised:
        rms = torch.sqrt(torch.mean(torch.square(waveform)))
        waveform = waveform*rms_normalised/rms
    if set_db:
        power = torch.mean(torch.square(waveform))
        normalising_factor = torch.sqrt(10 ** (set_db / 10) / power)
        waveform = waveform * normalising_factor
    if add_white_noise:
        waveform = add_noise_to_waveform(waveform, add_white_noise, 'white')
    if add_pink_noise:
        waveform = add_noise_to_waveform(waveform, add_pink_noise, 'pink')
    if add_brown_noise:
        waveform = add_noise_to_waveform(waveform, add_brown_noise, 'brown')

    if to_spec:
        if to_spec=='power':
            spec = torchaudio.transforms.Spectrogram(
                n_fft=specconfig['n_fft'],
                win_length=specconfig['win_length'], 
                hop_length=specconfig['hop_length'], 
                power=2.0
            )(waveform)
        elif to_spec=='energy':
            spec = torchaudio.transforms.Spectrogram(
                n_fft=2048, 
                win_length=2048, 
                hop_length=512, 
                power=None,
            )(waveform)
        return spec

    return waveform

def load_waveform(paths):
    waveforms=[]
    original_sample_rates=[]
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        waveform, original_sample_rate = torchaudio.load(path)
        waveforms.append(waveform)
        original_sample_rates.append(original_sample_rate)
    if len(waveforms)==1:
        return waveforms[0], original_sample_rates[0]
    return waveforms, original_sample_rates