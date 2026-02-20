# tests/recogniser/test-grid-recogniser-inference-localise.py

import os
import glob
import json
import struct
import yaml
from datetime import datetime, timedelta
from collections import defaultdict
import colorsys
from contextlib import redirect_stderr, redirect_stdout
from itertools import combinations
import logging
import copy
from geopy.distance import distance as geopy_distance
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from PIL import Image
from scipy.ndimage import center_of_mass, label
from scipy.signal import correlate
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from recogniser.spectrogram_tools import spectrogram_transformed
from colors import dusk_colormap
import pickle

# --- Metadata Extraction Components ---

def parse_guano_metadata(data: bytes) -> dict:
    """Parses a GUANO metadata block."""
    metadata = {}
    try:
        text = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        return {}
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip().replace('\\n', '\n')
    return metadata

def extract_guano_metadata_with_duration(file_path: str, sample_rate: int) -> dict:
    """Opens a WAV file, extracts GUANO metadata, and calculates duration."""
    metadata = {}
    try:
        with open(file_path, 'rb') as f:
            f.seek(4)
            # File size from RIFF header
            struct.unpack('<I', f.read(4))[0]
            f.seek(36)
            data_chunk_header = f.read(4)
            if data_chunk_header != b'data':
                f.seek(12)
                while f.read(4) != b'data':
                    sub_chunk_size = struct.unpack('<I', f.read(4))[0]
                    f.seek(sub_chunk_size, 1)
            
            data_size = struct.unpack('<I', f.read(4))[0]
            num_samples = data_size // 2 # Assuming 16-bit mono audio
            duration_seconds = num_samples / sample_rate
            metadata['duration_s'] = duration_seconds

            f.seek(0)
            if f.read(4) != b'RIFF' or f.read(4) == b'' or f.read(4) != b'WAVE':
                return metadata

            while True:
                chunk_id = f.read(4)
                if not chunk_id: break
                chunk_size_bytes = f.read(4)
                if not chunk_size_bytes: break
                chunk_size = struct.unpack('<I', chunk_size_bytes)[0]
                
                if chunk_id == b'guan':
                    chunk_data = f.read(chunk_size)
                    guano_meta = parse_guano_metadata(chunk_data)
                    metadata.update(guano_meta)
                    break
                else:
                    f.seek(chunk_size, 1)
                if chunk_size % 2 != 0:
                    f.seek(1, 1)
    except Exception as e:
        print(f"\nCould not process file {os.path.basename(file_path)}: {e}")
        return {}
    return metadata

def get_location_id(device_id: str, timestamp: datetime, location_mapping: dict) -> int | None:
    """Finds the correct locationID for a given device and timestamp."""
    if device_id not in location_mapping: return None
    device_dates = location_mapping[device_id]['dates_locationIDs']
    recording_mmdd = timestamp.strftime('%m%d')
    relevant_dates = [d for d in device_dates.keys() if d <= recording_mmdd]
    if not relevant_dates: return None
    latest_date = max(relevant_dates)
    return device_dates[latest_date]
    
def resample_log_mask_to_linear(log_space_mask, linear_spec_shape):
    """
    Resamples a mask from a logarithmic frequency scale to a linear frequency scale.
    Handles MPS compatibility by falling back to CPU for grid_sample.
    """
    # Check if we need to fallback to CPU (MPS often lacks full grid_sample support)
    original_device = log_space_mask.device
    use_cpu_fallback = (original_device.type == 'mps')
    
    target_device = torch.device('cpu') if use_cpu_fallback else original_device
    
    # Move input to target device
    log_space_mask = log_space_mask.to(target_device)
    
    linear_height, linear_width = linear_spec_shape
    
    y = torch.linspace(-1, 1, linear_height, device=target_device)
    x = torch.linspace(-1, 1, linear_width, device=target_device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    y_norm = (grid_y + 1) / 2
    log_y_norm = torch.log10(y_norm * 9 + 1)
    log_grid_y = log_y_norm * 2 - 1

    target_grid = torch.stack((grid_x, log_grid_y), dim=-1).unsqueeze(0)
    log_space_mask_unsqueezed = log_space_mask.float().unsqueeze(0).unsqueeze(0)
    
    # Perform grid sample
    linear_mask = F.grid_sample(
        log_space_mask_unsqueezed, 
        target_grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=False
    )
    
    result = linear_mask.squeeze(0).squeeze(0)
    
    # Move back if needed
    if use_cpu_fallback:
        result = result.to(original_device)
        
    return result

# --- Recogniser Components ---

class ClassifierHead(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(ClassifierHead, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_size = hidden_size
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier_layer = nn.Linear(last_size, output_size)
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier_layer(features)

def load_species_maps(filepath):
    df = pd.read_csv(filepath, header=None, names=['id', 'species_name'])
    id_to_species = {row.id: row.species_name for _, row in df.iterrows()}
    model_idx_to_species_name = {i: id_to_species[class_id] for i, class_id in enumerate(sorted(df.id.unique()))}
    print(f"\nLoaded {len(df['species_name'].unique())} species.")
    return model_idx_to_species_name

def get_complex_spectrogram(y, stft_params):
    if y.dim() == 1: y = y.unsqueeze(0)
    transform = torchaudio.transforms.Spectrogram(n_fft=stft_params['n_fft'], win_length=stft_params['win_length'], hop_length=stft_params['hop_length'], power=None)
    return transform(y)

def spec_to_image_for_model(complex_spec, resize_size):
    """
    Converts a complex spectrogram to a PIL Image using the external library.
    """
    power_spectrogram = torch.square(torch.abs(complex_spec))
    power_spectrogram = spectrogram_transformed(power_spectrogram, set_db=-10)
    
    image = spectrogram_transformed(
        power_spectrogram,
        to_pil=True,
        color_mode='RGB',
        log_scale=True,
        normalise='power_to_PCEN',
        resize=(resize_size, resize_size)
    )
    return image

def get_embedding_from_waveform(waveform_np, sample_rate, birdnet_analyzer, center_time_s=None, clip_duration_s=3):
    if center_time_s and clip_duration_s:
        center_sample = int(center_time_s * sample_rate)
        half_duration_samples = int((clip_duration_s / 2) * sample_rate)
        start_sample = max(0, center_sample - half_duration_samples)
        end_sample = min(len(waveform_np), center_sample + half_duration_samples)
        if start_sample >= end_sample: return None
        cropped_audio = waveform_np[start_sample:end_sample]
    else:
        cropped_audio = waveform_np
    
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            recording = RecordingBuffer(analyzer=birdnet_analyzer, buffer=cropped_audio, rate=sample_rate, min_conf=0.0)
            recording.extract_embeddings()
    if recording.embeddings:
        return np.mean([e['embeddings'] for e in recording.embeddings], axis=0) if len(recording.embeddings) > 1 else recording.embeddings[0]['embeddings']
    return None

# --- Visualization & Localization ---

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def generate_rainbow_colors(num_colors: int) -> list[str]:
    if not isinstance(num_colors, int) or num_colors <= 0: return []
    hex_colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb_float = colorsys.hls_to_rgb(hue, 0.5, 1.0)
        rgb_int = tuple(int(c * 255) for c in rgb_float)
        hex_colors.append(f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}".upper())
    return hex_colors

def extract_event_audio(grid_data_loc, start_time: datetime, end_time: datetime, sample_rate: int) -> torch.Tensor:
    """
    Extracts audio for a specific time range [start_time, end_time] from the file list 
    provided in grid_data_loc. Returns a single tensor.
    """
    duration_s = (end_time - start_time).total_seconds()
    num_samples = int(duration_s * sample_rate)
    output_waveform = torch.zeros(1, num_samples)

    # Find overlapping files
    overlapping_files = [f for f in grid_data_loc if f['start_time'] < end_time and f['end_time'] > start_time]
    
    for audio_file in overlapping_files:
        try:
            # Determine intersection
            intersect_start = max(start_time, audio_file['start_time'])
            intersect_end = min(end_time, audio_file['end_time'])
            
            if intersect_end <= intersect_start: continue

            # Calculate offsets in the File
            file_offset_start_s = (intersect_start - audio_file['start_time']).total_seconds()
            file_offset_end_s = (intersect_end - audio_file['start_time']).total_seconds()
            
            # Load specific segment (using torchaudio frame reading would be more efficient, 
            # but load+crop is safer given the existing structure)
            # Optimization: framing logic can be added here if files are huge
            file_waveform, sr = torchaudio.load(audio_file['path'])
            if sr != sample_rate:
                file_waveform = torchaudio.functional.resample(file_waveform, sr, sample_rate)

            f_start_sample = int(file_offset_start_s * sample_rate)
            f_end_sample = int(file_offset_end_s * sample_rate)
            
            # Handle bounds
            f_start_sample = max(0, f_start_sample)
            f_end_sample = min(file_waveform.shape[1], f_end_sample)
            
            segment = file_waveform[:, f_start_sample:f_end_sample]

            # Calculate offsets in the Output Tensor
            out_offset_start_s = (intersect_start - start_time).total_seconds()
            out_start_sample = int(out_offset_start_s * sample_rate)
            out_end_sample = out_start_sample + segment.shape[1]
            
            if out_end_sample > num_samples:
                # Clip if rounding errors push us over
                diff = out_end_sample - num_samples
                segment = segment[:, :-diff]
                out_end_sample = num_samples

            output_waveform[:, out_start_sample:out_end_sample] = segment

        except Exception as e:
            print(f"Error loading event audio from {os.path.basename(audio_file['path'])}: {e}")

    return output_waveform

def load_sensor_locations(csv_path: str) -> tuple[dict, pd.DataFrame]:
    """
    Loads lat/long from CSV and converts to local Cartesian (meters).
    Returns a dict {loc_id: (x, y)} and a DataFrame for plotting.
    """
    df = pd.read_csv(csv_path)
    
    # Set the center of the coordinate system to the mean lat/long
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    center_point = (center_lat, center_lon)
    
    local_coords = {}
    
    for _, row in df.iterrows():
        # Calculate Y (North/South distance)
        # Point with same longitude, different latitude
        dist_y = geopy_distance(center_point, (row['latitude'], center_lon)).meters
        if row['latitude'] < center_lat: dist_y = -dist_y
        
        # Calculate X (East/West distance)
        # Point with same latitude, different longitude
        dist_x = geopy_distance(center_point, (center_lat, row['longitude'])).meters
        if row['longitude'] < center_lon: dist_x = -dist_x
        
        local_coords[int(row['locationID'])] = np.array([dist_x, dist_y])
        
    # Update DF with calculated meters for debug plotting
    df['x_m'] = df['locationID'].map(lambda x: local_coords[x][0])
    df['y_m'] = df['locationID'].map(lambda x: local_coords[x][1])
    
    return local_coords, df

def reconstruct_audio_for_localization(full_clip, det, event_start_time, stft_params, device):
    """
    1. Computes STFT of the full_clip (duration of the group event).
    2. Takes the detection mask (cropped to vocalisation).
    3. Resamples mask (Log -> Linear) to match frequency bins.
    4. Pastes the mask into the full spectrogram at the correct time index.
    5. Performs ISTFT to isolate audio.
    """
    # 1. Compute Complex Spectrogram of the Full Event
    transform = torchaudio.transforms.Spectrogram(
        n_fft=stft_params['n_fft'], 
        win_length=stft_params['win_length'], 
        hop_length=stft_params['hop_length'], 
        power=None
    ).to(device)
    
    clip_t = full_clip.to(device)
    if clip_t.dim() == 1: clip_t = clip_t.unsqueeze(0)
    
    complex_spec = transform(clip_t).squeeze(0) # (Freq_Linear, Time_Frames)
    
    # 2. Prepare the Canvas Mask (Same size as spectrogram, initialized to zeros)
    canvas_mask = torch.zeros(complex_spec.shape, device=device)
    
    # 3. Process the Detection Mask
    # det['mask'] is (Freq_Log, Time_Pixels)
    mask_np = det['mask']
    mask_t = torch.from_numpy(mask_np).float().to(device)
    mask_t = torch.flip(mask_t, dims=[0]) # Flip vertically
    
    # Calculate Target Dimensions for the Slice
    # We want to convert Log Freq -> Linear Freq (513 bins usually)
    # We Keep the width (Time) consistent with the input mask
    target_freq_bins = complex_spec.shape[0]
    target_width_pixels = mask_np.shape[1]
    
    # Resample ONLY the slice
    # This correctly handles the Log->Linear stretch for just the vocalisation content
    linear_mask_slice = resample_log_mask_to_linear(mask_t, (target_freq_bins, target_width_pixels))
    
    # Thresholding
    upscaled_mask_slice = (linear_mask_slice > 0.5).float()
    
    # 4. Calculate Insertion Position
    # Map absolute time to spectrogram frames
    sample_rate = 48000 # Passed or global, usually standard
    # Time per frame in the spectrogram
    seconds_per_frame = stft_params['hop_length'] / sample_rate 
    
    # Start time of the specific detection vs Start time of the whole event clip
    time_offset_s = (det['absolute_start'] - event_start_time).total_seconds()
    
    start_frame = int(time_offset_s / seconds_per_frame)
    end_frame = start_frame + upscaled_mask_slice.shape[1]
    
    # Bounds checking
    start_frame = max(0, start_frame)
    end_frame = min(canvas_mask.shape[1], end_frame)
    
    # Determine width to paste (handle edge clipping)
    paste_width = end_frame - start_frame
    
    if paste_width > 0:
        # Paste the slice into the canvas
        canvas_mask[:, start_frame:end_frame] = upscaled_mask_slice[:, :paste_width]
    
    # 5. Apply Mask & ISTFT
    masked_spec = complex_spec * canvas_mask
    
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=stft_params['n_fft'], 
        win_length=stft_params['win_length'], 
        hop_length=stft_params['hop_length']
    ).to(device)
    
    isolated_audio = istft(masked_spec.unsqueeze(0)).squeeze(0)

    # The resulting audio is the length of the event, with only the vocalisation present
    return isolated_audio.cpu()

def define_search_grid(mic_coords, grid_size, resolution=1):
    """Pre-calculates the grid points for the search area."""
    coords = np.array(list(mic_coords.values()))
    center = coords.mean(axis=0)
    
    width, height = grid_size
    min_x, max_x = center[0] - width / 2, center[0] + width / 2
    min_y, max_y = center[1] - height / 2, center[1] + height / 2
    
    x_range = np.arange(min_x, max_x, resolution)
    y_range = np.arange(min_y, max_y, resolution)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    return grid_x, grid_y, grid_points

def solve_location_grid_search(mic_coords, tdoas, ref_idx, grid_points):
    """
    Performs grid search to find source location for a specific set of TDOAs.
    Uses pre-calculated grid_points for efficiency.
    """
    ref_pos = mic_coords[ref_idx]
    c = 343.0 
    
    total_error = np.zeros(len(grid_points))
    
    # Calculate error for each mic pair (ref vs i) within this specific TDOA set
    for loc_id, tdoa_measured in tdoas.items():
        if loc_id == ref_idx: continue
        
        mic_pos = mic_coords[loc_id]
        
        # Theoretical distances
        dist_to_ref = np.linalg.norm(grid_points - ref_pos, axis=1)
        dist_to_mic = np.linalg.norm(grid_points - mic_pos, axis=1)
        
        # Theoretical DDOA
        ddoa_theoretical = dist_to_mic - dist_to_ref
        ddoa_measured = tdoa_measured * c 
        
        error = (ddoa_theoretical - ddoa_measured) ** 2
        total_error += error

    # Find Min
    min_idx = np.argmin(total_error)
    est_pos = grid_points[min_idx]
    min_err = total_error[min_idx]
    
    # Return total_error array as well
    return est_pos, min_err, total_error

def compute_gaussian_mle_grid(estimates, grid_x, grid_y, sigma=4.0):
    """
    Combines individual location estimates using Gaussian Maximum Likelihood Estimation.
    Calculates the likelihood surface: sum of gaussians centered at each estimate.
    """
    likelihood_surface = np.zeros_like(grid_x, dtype=np.float64)
    
    # Pre-compute constant factor (optional for argmax but good for correctness)
    norm_factor = 1.0 / (np.sqrt(2 * np.pi * sigma**2))
    
    for (est_x, est_y) in estimates:
        # Vectorized distance calculation over the grid
        dist_sq = (grid_x - est_x)**2 + (grid_y - est_y)**2
        gauss = norm_factor * np.exp(-dist_sq / (2 * sigma**2))
        likelihood_surface += gauss
        
    return likelihood_surface

def compute_pairwise_lag(sig1_tensor, sig2_tensor, valid_range1, valid_range2, sample_rate, max_tdoa_s):
    """
    Computes lag of sig2 relative to sig1.
    Assumes inputs are masked ISTFTs (mostly zeros).
    Uses Max-Abs normalization to preserve zero-padding.
    """
    # 1. Determine the intersection of valid data
    max_lag_samples = int(np.ceil(max_tdoa_s * sample_rate))
    inter_start = max(valid_range1[0], valid_range2[0])
    inter_end = min(valid_range1[1], valid_range2[1])
    
    if inter_end <= inter_start:
        return None, 0.0, None, None
    
    s1 = sig1_tensor.numpy()[inter_start:inter_end]
    s2 = sig2_tensor.numpy()[inter_start:inter_end]

    # --- NORMALIZATION (Max-Abs) ---
    # Since the signals are sparse (mostly zeros), we must NOT subtract the mean.
    # Subtracting the mean would turn the zeros into non-zeros, causing
    # artifacts when we pad with hard zeros later.
    # We simply scale them to -1.0 to 1.0 range.
    s1 = s1 / (np.max(np.abs(s1)) + 1e-9)
    s2 = s2 / (np.max(np.abs(s2)) + 1e-9)

    # 2. Pad ONLY s2 (the "search area")
    # We pad s2 with zeros so s1 can slide 'off the edge' by max_lag_samples
    pad_width = (max_lag_samples, max_lag_samples)
    s2_padded = np.pad(s2, pad_width, mode='constant', constant_values=0)
    
    # 3. Correlate using mode='valid'
    # 'valid' slides s1 over s2_padded.
    # Since s1 has zeros at the edges (from the mask) and s2_padded has zeros 
    # at the edges (from padding), this handles the boundaries naturally.
    corr = correlate(s2_padded, s1, mode='valid', method='fft')
 
    # 4. Map index to time lag
    # The result array indices [0, 1, ..., 2*max] map to lags [-max, ..., +max]
    lags_axis_samples = np.arange(len(corr)) - max_lag_samples
    lags_seconds_axis = lags_axis_samples / sample_rate

    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]
    
    calculated_lag = lags_seconds_axis[peak_idx]

    return calculated_lag, peak_val, corr, lags_seconds_axis

def merge_detections_for_localization(dets: list) -> dict:
    """Merges multiple detections at a single location into one composite detection."""
    if not dets: return None
    if len(dets) == 1: return dets[0]

    print(f'MERGING {len(dets)} DETECTIONS FOR LOCALIZATION')
    
    # Sort by start time to establish the timeline
    dets.sort(key=lambda x: x['absolute_start'])
    first, last = dets[0], dets[-1]
    
    # Calculate canvas size
    px_per_sec = first.get('px_per_sec', 22.4)
    total_duration_s = (last['absolute_end'] - first['absolute_start']).total_seconds()
    
    # Width is time difference in pixels + width of the last mask
    # We use round() to minimize float drift when converting seconds to pixels
    total_width = int(round(total_duration_s * px_per_sec)) + last['mask'].shape[1]
    height = first['mask'].shape[0]
    
    # Create composite mask
    new_mask = np.zeros((height, total_width), dtype=first['mask'].dtype)
    start_ref = first['absolute_start']
    
    for d in dets:
        offset_s = (d['absolute_start'] - start_ref).total_seconds()
        offset_px = int(round(offset_s * px_per_sec))
        
        # Determine paste region (handle potential boundary overshoots)
        mask_w = d['mask'].shape[1]
        end_px = min(offset_px + mask_w, total_width)
        paste_w = end_px - offset_px
        
        if paste_w > 0:
            new_mask[:, offset_px:end_px] = np.maximum(
                new_mask[:, offset_px:end_px], 
                d['mask'][:, :paste_w]
            )

    # Recalculate metadata for the merged object
    cy, cx = center_of_mass(new_mask)
    merged = first.copy()
    merged.update({
        'mask': new_mask,
        'absolute_end': last['absolute_end'],
        'confidence': max(d['confidence'] for d in dets),
        'center_of_mass': (cy, cx)
    })
    return merged

def visualize_triplet_detail(triplet_ids, group_dets, raw_clips_dict, core_signals, 
                             global_bounds, window_start, mic_coords, est_pos, 
                             calculated_lags, sample_rate, error_score, cycle_error, valid_windows,
                             pairwise_debug_data, blind_offset_tolerance, grid_size, grid_info=None):
    """
    Detailed debug plot. Consolidated into one figure.
    Layout: [Raw Spec] | [Cross-Correlation] | [Mask Alignment] | [Localization Map]
    """
    m1, m2, m3 = triplet_ids
    lag_12, lag_23, lag_13 = calculated_lags
    start_sample, end_sample = global_bounds
    SPEED_OF_SOUND = 343.0
    
    # Create Figure
    fig = plt.figure(figsize=(24, 10)) 
    fig.suptitle(f"Localization Triplet Detail (Error: {error_score:.4f} | Cycle Err: {cycle_error:.4f}s)", fontsize=14)
    
    # 3 Rows (one per mic/pair), 4 Columns
    # Width ratios favor the map slightly
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1.5])

    # --- COLUMN 0: Raw Spectrograms (Single Mics) ---
    mics = [m1, m2, m3]
    for i, mid in enumerate(mics):
        ax = fig.add_subplot(gs[i, 0])
        
        raw_full = raw_clips_dict[mid]
        sig_full_np = raw_full.numpy()
        sig_full_np += np.random.normal(0, 1e-10, sig_full_np.shape) # Add epsilon noise to prevent log(0) warnings
        clip_duration_s = len(sig_full_np) / sample_rate
        
        ax.specgram(sig_full_np, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='gray', xextent=[0, clip_duration_s])
        
        # Valid region lines
        v_start_idx, v_end_idx = valid_windows[mid]
        ax.axvline(v_start_idx / sample_rate, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(v_end_idx / sample_rate, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)

        # Overlay Detection Mask
        det = next((d for d in group_dets if d['loc_id'] == mid), None)
        if det:
            t_det_start = (det['absolute_start'] - window_start).total_seconds()
            t_det_end = (det['absolute_end'] - window_start).total_seconds()
            if t_det_end > 0 and t_det_start < clip_duration_s:
                rel_t_start = max(0, t_det_start)
                rel_t_end = min(clip_duration_s, t_det_end)
                mask_t = torch.from_numpy(det['mask']).float()
                mask_t = torch.flip(mask_t, dims=[0])
                linear_mask = resample_log_mask_to_linear(mask_t, (513, det['mask'].shape[1]))
                lin_mask_np = linear_mask.numpy()
                
                # Draw Box
                lin_rows, _ = np.where(lin_mask_np > 0.1)
                if len(lin_rows) > 0:
                    f_min = (lin_rows.min() / 513) * (sample_rate / 2)
                    f_max = (lin_rows.max() / 513) * (sample_rate / 2)
                    rect = patches.Rectangle((rel_t_start, f_min), rel_t_end - rel_t_start, f_max - f_min,
                                             linewidth=2, edgecolor='lime', facecolor='none', alpha=0.9)
                    ax.add_patch(rect)
                
                # Draw Mask Fill
                extent = [rel_t_start, rel_t_end, 0, sample_rate / 2]
                overlay_img = np.zeros((lin_mask_np.shape[0], lin_mask_np.shape[1], 4))
                overlay_img[..., 0] = 1.0 # Red
                overlay_img[..., 3] = lin_mask_np * 0.4 # Alpha
                ax.imshow(overlay_img, origin='lower', extent=extent, aspect='auto')

        ax.set_ylabel(f"Freq (Hz)")
        ax.text(0.02, 0.9, f"Mic {mid}", transform=ax.transAxes, color='white', fontweight='bold', 
                bbox=dict(facecolor='black', alpha=0.7))
        ax.set_xlim(0, clip_duration_s)
        if i < 2: ax.set_xticklabels([])

    # Process Pairwise Logic for Columns 1 and 2
    pairs = [(m1, m2, lag_12), (m2, m3, lag_23), (m1, m3, lag_13)]
    pair_labels = [f"M{m1}->M{m2}", f"M{m2}->M{m3}", f"M{m1}->M{m3}"]

    for i, ((ma, mb, used_lag), label) in enumerate(zip(pairs, pair_labels)):
        
        # --- COLUMN 1: Cross-Correlation Plots (Pairwise) ---
        ax_corr = fig.add_subplot(gs[i, 1])
        
        key = tuple(sorted((ma, mb)))
        if key not in pairwise_debug_data:
            ax_corr.text(0.5, 0.5, "Data Missing", ha='center', va='center')
        else:
            corr_data, lags_sec = pairwise_debug_data[key]
            # Handle flip if order is reversed in tuple key
            if ma > mb:
                corr_plot = np.flip(corr_data)
                lags_plot = -np.flip(lags_sec)
            else:
                corr_plot = corr_data
                lags_plot = lags_sec

            ax_corr.plot(lags_plot, corr_plot, color='cyan', lw=1)
            ax_corr.axvline(used_lag, color='red', linestyle='--', linewidth=1.5, label=f'Lag: {used_lag*1000:.1f}ms')
            
            p1, p2 = mic_coords[ma], mic_coords[mb]
            dist = np.linalg.norm(p1 - p2)
            max_theoretical_lag = ((dist / SPEED_OF_SOUND) * 1.2) + blind_offset_tolerance
            
            ax_corr.set_xlim(-max_theoretical_lag, max_theoretical_lag)
            ax_corr.legend(loc='upper right', fontsize='x-small')
            ax_corr.set_title(f"Corr {label} (Dist: {dist:.1f}m)", fontsize=10, color='white', backgroundcolor='black')
            ax_corr.grid(True, alpha=0.3)
            ax_corr.set_yticklabels([])

        # --- COLUMN 2: Mask Alignment (Pairwise) ---
        ax_mask = fig.add_subplot(gs[i, 2])
        
        det_a = next((d for d in group_dets if d['loc_id'] == ma), None)
        det_b = next((d for d in group_dets if d['loc_id'] == mb), None)
        
        if det_a and det_b:
            # Helper to get linear mask for plotting
            def get_lin_mask(d):
                mt = torch.from_numpy(d['mask']).float()
                mt = torch.flip(mt, dims=[0])
                lm = resample_log_mask_to_linear(mt, (513, d['mask'].shape[1]))
                return lm.numpy()

            mask_a = get_lin_mask(det_a)
            mask_b = get_lin_mask(det_b)
            
            t_start_a = (det_a['absolute_start'] - window_start).total_seconds()
            t_end_a = (det_a['absolute_end'] - window_start).total_seconds()
            t_start_b = (det_b['absolute_start'] - window_start).total_seconds()
            t_end_b = (det_b['absolute_end'] - window_start).total_seconds()
            
            shifted_start_b = t_start_b - used_lag
            shifted_end_b = t_end_b - used_lag
            
            # Plot A (Blue)
            extent_a = [t_start_a, t_end_a, 0, sample_rate/2]
            img_a = np.zeros((mask_a.shape[0], mask_a.shape[1], 4))
            img_a[..., 2] = 1.0 
            img_a[..., 3] = mask_a * 0.6 
            ax_mask.imshow(img_a, origin='lower', extent=extent_a, aspect='auto')
            
            # Plot B (Red, Shifted)
            extent_b = [shifted_start_b, shifted_end_b, 0, sample_rate/2]
            img_b = np.zeros((mask_b.shape[0], mask_b.shape[1], 4))
            img_b[..., 0] = 1.0 
            img_b[..., 3] = mask_b * 0.6 
            ax_mask.imshow(img_b, origin='lower', extent=extent_b, aspect='auto')
            
            # Zoom to union
            union_min = min(t_start_a, shifted_start_b)
            union_max = max(t_end_a, shifted_end_b)
            pad_t = (union_max - union_min) * 0.2
            ax_mask.set_xlim(union_min - pad_t, union_max + pad_t)

            # Zoom freq
            rows_a, _ = np.where(mask_a > 0.1)
            rows_b, _ = np.where(mask_b > 0.1)
            if len(rows_a) > 0 and len(rows_b) > 0:
                y_min_idx = min(rows_a.min(), rows_b.min())
                y_max_idx = max(rows_a.max(), rows_b.max())
                f_min = (y_min_idx / 513) * (sample_rate / 2)
                f_max = (y_max_idx / 513) * (sample_rate / 2)
                pad_f = (f_max - f_min) * 0.2
                ax_mask.set_ylim(max(0, f_min - pad_f), min(sample_rate/2, f_max + pad_f))

            ax_mask.text(0.05, 0.9, f"Ref: M{ma} (Blu)\nShift: M{mb} (Red)", transform=ax_mask.transAxes, 
                         color='white', fontsize=8, va='top', bbox=dict(facecolor='black', alpha=0.5))
        
        ax_mask.set_yticklabels([])
        ax_mask.grid(True, alpha=0.3, linestyle=':')

    # --- COLUMN 3: Localization Map (Spanning all rows) ---
    ax_map = fig.add_subplot(gs[:, 3])
    
    if grid_info is not None:
        grid_x, grid_y, error_grid = grid_info
        pseudo_likelihood = 1.0 / (error_grid + 1e-3)
        ax_map.contourf(grid_x, grid_y, pseudo_likelihood, levels=50, cmap='inferno')

    # Plot All Mics
    all_x = [pos[0] for pos in mic_coords.values()]
    all_y = [pos[1] for pos in mic_coords.values()]
    ax_map.scatter(all_x, all_y, c='lightgray', s=50, label='Inactive')
    
    # Plot Active Triplet
    active_coords = np.array([mic_coords[mid] for mid in triplet_ids])
    ax_map.scatter(active_coords[:, 0], active_coords[:, 1], c=['red', 'lime', 'blue'], s=150, edgecolors='black', zorder=5)
    
    # Labels
    for mid, color in zip(triplet_ids, ['red', 'lime', 'blue']):
        ax_map.text(mic_coords[mid][0], mic_coords[mid][1]+1, f"M{mid}", color='black', fontweight='bold', fontsize=12)
    
    # Plot Estimate
    ax_map.scatter(est_pos[0], est_pos[1], c='gold', marker='*', s=400, edgecolors='black', label='Source Est', zorder=10)
    
    ax_map.set_aspect('equal')
    ax_map.grid(True, linestyle=':', alpha=0.5)
    
    # Set Map Bounds
    if grid_info is not None:
        grid_x, grid_y, _ = grid_info
        ax_map.set_xlim(grid_x.min(), grid_x.max())
        ax_map.set_ylim(grid_y.min(), grid_y.max())
    else:
        all_xs = [pos[0] for pos in mic_coords.values()] + [est_pos[0]]
        all_ys = [pos[1] for pos in mic_coords.values()] + [est_pos[1]]
        
        center_x, center_y = np.mean(all_xs), np.mean(all_ys)
        width, height = grid_size
        ax_map.set_xlim(center_x - width/2, center_x + width/2)
        ax_map.set_ylim(center_y - height/2, center_y + height/2)

    plt.tight_layout()
    plt.show()

def visualize_group_debug(group_dets, grid_data, sample_rate):
    """
    Debug plot: 3x3 grid of spectrograms for the group event with masks overlaid.
    """
    if not group_dets: return

    # 1. Determine Global Bounds for the plot
    # Add padding to context
    pad = timedelta(seconds=0.5)
    min_start = min(d['absolute_start'] for d in group_dets) - pad
    max_end = max(d['absolute_end'] for d in group_dets) + pad
    duration_s = (max_end - min_start).total_seconds()
    
    species = group_dets[0]['species_name']
    gid = group_dets[0].get('group_id', '?')

    # 2. Setup Plot
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    fig.suptitle(f"Debug Group {gid}: {species} | {min_start.strftime('%H:%M:%S')}", fontsize=14)
    
    # 3. Iterate Locations 1-9
    for i in range(3):
        for j in range(3):
            loc_id = (i * 3) + j + 1 # Maps 1..9
            ax = axes[i, j]
            
            # A. Get Audio
            files = grid_data.get(loc_id, [])
            if not files:
                ax.text(0.5, 0.5, "No Files", ha='center', va='center')
                ax.set_axis_off()
                continue
                
            waveform = extract_event_audio(files, min_start, max_end, sample_rate)
            sig_np = waveform.squeeze().numpy()
            
            # B. Plot Spectrogram
            if np.sum(np.abs(sig_np)) == 0:
                ax.text(0.5, 0.5, "Silence/Gap", ha='center', va='center')
            else:
                sig_np += np.random.normal(0, 1e-10, sig_np.shape) # Add epsilon noise to prevent log(0) warnings
                Pxx, freqs, bins, im = ax.specgram(
                    sig_np, 
                    NFFT=1024, 
                    Fs=sample_rate, 
                    noverlap=512,
                    cmap='gray',
                    xextent=[0, duration_s]
                )
            
            # C. Overlay Detection Mask (if exists)
            det = next((d for d in group_dets if d['loc_id'] == loc_id), None)
            
            if det:
                t_det_start = (det['absolute_start'] - min_start).total_seconds()
                t_det_end = (det['absolute_end'] - min_start).total_seconds()
                
                # 1. Prepare Mask (Log -> Linear)
                # Note: We use existing helper resample_log_mask_to_linear
                mask_t = torch.from_numpy(det['mask']).float()
                mask_t = torch.flip(mask_t, dims=[0]) # Flip vertically
                
                # Resample to 513 bins (0-Nyquist)
                linear_mask = resample_log_mask_to_linear(mask_t, (513, det['mask'].shape[1]))
                lin_mask_np = linear_mask.numpy()
                
                # 2. Create Overlay Image (Red with Alpha)
                overlay = np.zeros((lin_mask_np.shape[0], lin_mask_np.shape[1], 4))
                overlay[..., 0] = 1.0 # Red
                overlay[..., 3] = lin_mask_np * 0.6 # Alpha
                
                # 3. Plot using extent
                extent = [t_det_start, t_det_end, 0, sample_rate / 2]
                ax.imshow(overlay, origin='lower', extent=extent, aspect='auto')
                
                # Box
                rect = patches.Rectangle(
                    (t_det_start, 0), 
                    t_det_end - t_det_start, 
                    sample_rate / 2, 
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Label
                ax.text(
                    t_det_start, (sample_rate/2) * 0.9, 
                    f"{det['confidence']:.2f}", 
                    color='lime', fontweight='bold', fontsize=8, 
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

            # Styling
            ax.set_title(f"Loc {loc_id}", fontsize=10, fontweight='bold', pad=2)
            ax.set_xlim(0, duration_s)
            ax.set_ylim(0, sample_rate / 2)
            
            if i < 2: ax.set_xticklabels([]) 
            if j > 0: ax.set_yticklabels([]) 

    plt.tight_layout()
    plt.show()

def attempt_localization_on_group(
    group_dets, 
    mic_coords, 
    sample_rate, 
    stft_params, 
    device,
    grid_data,
    blind_offset_tolerance,
    localisation_config,
    plot=True
):
    """
    Localises a 'Completed' group. 
    1. Determines global start/end of the group event.
    2. Loads specific audio for that duration.
    3. Reconstructs isolated audio using the exact mask.
    4. Correlates.
    """
    if len(group_dets) < 3: return
    
    # 1. Determine Event Bounds
    # Add padding to ensure the vocalisation isn't right on the edge of the STFT window
    padding_s = 0.5 
    min_start = min(d['absolute_start'] for d in group_dets) - timedelta(seconds=padding_s)
    max_end = max(d['absolute_end'] for d in group_dets) + timedelta(seconds=padding_s)
    
    # For visualization/debug passed to triplet function
    vis_start_idx = 0
    vis_end_idx = int((max_end - min_start).total_seconds() * sample_rate)

    core_signals = {}  # Isolated audio
    raw_signals = {}   # Raw audio (for debug/vis)
    valid_windows = {} # Valid ranges
    
    print(f"   -> Processing Group Event: {min_start.strftime('%H:%M:%S.%f')} - {max_end.strftime('%H:%M:%S.%f')}")

    for det in group_dets:
        loc_id = det['loc_id']
        files = grid_data.get(loc_id, [])
        if not files: continue

        # Extract Raw Audio for the Event Duration
        full_clip = extract_event_audio(files, min_start, max_end, sample_rate)
        
        # Check if we actually got audio (not just zeros due to missing files)
        if full_clip.abs().sum() == 0:
            continue
            
        raw_signals[loc_id] = full_clip.squeeze()
        
        # In this event-based approach, the "Valid Window" is the whole clip 
        # because we specifically requested existing data. 
        # (Though technically extract_event_audio pads with zeros if file missing, 
        # but for simplicity we assume valid for now or add complex checking later)
        valid_windows[loc_id] = (0, full_clip.shape[1])

        # Reconstruct Isolated Audio
        try:
            iso_audio = reconstruct_audio_for_localization(full_clip, det, min_start, stft_params, device)
            
            # Ensure length matching (ISTFT can differ by small frames)
            if iso_audio.shape[0] != full_clip.shape[1]:
                target_len = full_clip.shape[1]
                if iso_audio.shape[0] > target_len:
                    iso_audio = iso_audio[:target_len]
                else:
                    pad = torch.zeros(target_len - iso_audio.shape[0])
                    iso_audio = torch.cat([iso_audio, pad])
                    
            core_signals[loc_id] = iso_audio
        except Exception as e:
            print(f"Error reconstructing audio for Loc {loc_id}: {e}")
            continue

    if len(core_signals) < 3: 
        print("   -> Not enough valid signals extracted.")
        return

    # 2. Pairwise Lags & Triplet Solving
    
    mic_ids = list(core_signals.keys())
    SPEED_OF_SOUND = 343.0
    pairwise_data = {} 
    pairwise_debug_data = {} 
    tdoa_sums = defaultdict(float) 
    tdoa_counts = defaultdict(int)

    # Calculate Max TDOA based on max physical distance
    # (Since we are now in a custom length window, max_lag_s is critical)
    all_coords = [mic_coords[m] for m in mic_ids]
    max_physical_dist = 0
    for i in range(len(all_coords)):
        for j in range(i+1, len(all_coords)):
            d = np.linalg.norm(all_coords[i] - all_coords[j])
            if d > max_physical_dist: max_physical_dist = d
            
    max_lag_s = ((max_physical_dist / SPEED_OF_SOUND) * 1.2) + blind_offset_tolerance

    for m1, m2 in combinations(mic_ids, 2):
        key = tuple(sorted((m1, m2)))
        
        lag, val, corr_arr, lags_axis = compute_pairwise_lag(
            core_signals[key[0]], 
            core_signals[key[1]], 
            valid_windows[key[0]],
            valid_windows[key[1]],
            sample_rate, 
            max_lag_s
        )
        
        if lag is not None:
            pairwise_data[key] = (lag, val)
            # Store for plotting
            pairwise_debug_data[key] = (corr_arr, lags_axis)
            
            tdoa_sums[key[0]] += lag
            tdoa_counts[key[0]] += 1
            
            tdoa_sums[key[1]] += (-lag)
            tdoa_counts[key[1]] += 1
            
    # Compute Average TDOA per mic
    avg_tdoa_offsets = {}
    for mid in mic_ids:
        if tdoa_counts[mid] > 0:
            avg_tdoa_offsets[mid] = tdoa_sums[mid] / tdoa_counts[mid]
        else:
            avg_tdoa_offsets[mid] = 0.0

    # 3. Triplet Combinations & Processing using Pre-calculated data
    triplets = list(combinations(mic_ids, 3))
    print(f"  > Processing {len(triplets)} triplet combinations...")
    
    grid_x, grid_y, grid_points = define_search_grid(mic_coords, localisation_config['grid_size'])
    triplet_results = [] 
    consistency_threshold_s = localisation_config['consistency_threshold']
    
    vis_start = max(v[0] for v in valid_windows.values())
    vis_end = min(v[1] for v in valid_windows.values())

    for triplet in triplets:
        m1, m2, m3 = triplet
        
        pair_12 = tuple(sorted((m1, m2)))
        pair_13 = tuple(sorted((m1, m3)))
        pair_23 = tuple(sorted((m2, m3)))
        
        if pair_12 not in pairwise_data or pair_13 not in pairwise_data or pair_23 not in pairwise_data:
            continue
            
        l12, v12 = pairwise_data[pair_12]
        l13, v13 = pairwise_data[pair_13]
        l23, v23 = pairwise_data[pair_23]
        
        lag_12 = l12 if m1 == pair_12[0] else -l12
        lag_13 = l13 if m1 == pair_13[0] else -l13
        lag_23 = l23 if m2 == pair_23[0] else -l23

        # 1. Consistency Check (Triangle Equality)
        cycle_error = abs((lag_12 + lag_23) - lag_13)
        
        if cycle_error > consistency_threshold_s:
            continue
            
        # 2. Solve
        sub_tdoas = { m1: 0.0, m2: lag_12, m3: lag_13 }
        est_pos, min_error, error_flat = solve_location_grid_search(mic_coords, sub_tdoas, m1, grid_points)
        
        # Reshape flat error array back to 2D grid shape
        error_grid = error_flat.reshape(grid_x.shape)

        # 3. Optional Detailed Visualization per Triplet
        # visualize_triplet_detail(
        #     triplet_ids=triplet,
        #     group_dets=group_dets,
        #     raw_clips_dict=raw_signals,    
        #     core_signals=core_signals,     
        #     global_bounds=(vis_start, vis_end),
        #     window_start=min_start,
        #     mic_coords=mic_coords,
        #     est_pos=est_pos,
        #     calculated_lags=(lag_12, lag_23, lag_13),
        #     sample_rate=sample_rate,
        #     error_score=min_error,
        #     cycle_error=cycle_error,
        #     valid_windows=valid_windows,
        #     pairwise_debug_data=pairwise_debug_data,
        #     blind_offset_tolerance=blind_offset_tolerance,
        #     grid_size=localisation_config['grid_size'],
        #     grid_info=(grid_x, grid_y, error_grid)
        # )

        weight = (v12 + v13 + v23) / 3.0
        triplet_results.append({'ids': triplet, 'pos': est_pos, 'error': min_error, 'weight': weight})

    if not triplet_results:
        print("  > No valid triplets found (consistency check or grid search failed).")
        return

    # 4. Gaussian MLE Aggregation
    estimates_coords = [r['pos'] for r in triplet_results]
    likelihood_grid = compute_gaussian_mle_grid(estimates_coords, grid_x, grid_y, sigma=100)
    max_idx = np.unravel_index(np.argmax(likelihood_grid), likelihood_grid.shape)
    final_x = grid_x[max_idx]
    final_y = grid_y[max_idx]
    final_pos = np.array([final_x, final_y])

    # 5. Plotting
    if plot:
        ref_det = group_dets[0]
        species = ref_det['species_name']
        ts = ref_det['absolute_time'].strftime('%H:%M:%S')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.contourf(grid_x, grid_y, likelihood_grid, levels=50, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Likelihood')
        
        active_ids = list(core_signals.keys())
        ax_x = [mic_coords[lid][0] for lid in active_ids]
        ax_y = [mic_coords[lid][1] for lid in active_ids]
        ax.scatter(ax_x, ax_y, c='red', edgecolors='black', s=120, marker='^', label='Active Mic', zorder=6)
        
        for lid in active_ids:
            x, y = mic_coords[lid]
            offset = avg_tdoa_offsets.get(lid, 0.0)
            sign = "+" if offset >= 0 else ""
            label_text = f"Loc {lid}\nAvg: {sign}{offset*1000:.1f}ms"
            ax.text(x, y + 2, label_text, color='white', fontsize=9, ha='center', va='bottom', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1))
        
        inactive_ids = [lid for lid in mic_coords.keys() if lid not in active_ids]
        if inactive_ids:
            inax_x = [mic_coords[lid][0] for lid in inactive_ids]
            inax_y = [mic_coords[lid][1] for lid in inactive_ids]
            ax.scatter(inax_x, inax_y, c='gray', edgecolors='black', s=100, marker='^', label='Inactive Mic', zorder=5, alpha=0.5)

        for res in triplet_results:
            tx, ty = res['pos']
            ax.scatter(tx, ty, c='orange', s=30, alpha=0.6, marker='x', zorder=8)

        ax.scatter(final_x, final_y, c='white', edgecolors='black', s=300, marker='*', label='MLE Source', zorder=10)
        
        ax.set_title(f"Localization: {species} @ {ts} | Group G{ref_det.get('group_id')} | {len(triplet_results)} Consistent Triplets")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        all_xs = [c[0] for c in mic_coords.values()] + [p['pos'][0] for p in triplet_results] + [final_x]
        all_ys = [c[1] for c in mic_coords.values()] + [p['pos'][1] for p in triplet_results] + [final_y]
        margin = 30
        ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
        ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
        
        plt.show()

# --- Grid Initialization and Processing Logic ---

def initialize_audio_grid(grid_dir: str, mapping_file: str, blind_synchronisation_offsets_path: str, sample_rate: int):
    print("--- Initializing Audio Grid ---")
    try:
        with open(mapping_file, 'r') as f: mic_to_loc_id = json.load(f)
    except Exception as e:
        print(f"FATAL: Could not load or parse mapping file '{mapping_file}': {e}")
        return None, None, None

    try:
        with open(blind_synchronisation_offsets_path, 'r') as f: blind_sync_offsets = json.load(f)
    except Exception as e:
        print(f"FATAL: Could not load or parse blind synchronisation offsets file '{blind_synchronisation_offsets_path}': {e}")
        return None, None, None

    wav_files = sorted(list(set(glob.glob(os.path.join(grid_dir, '**', '*.wav'), recursive=True) + glob.glob(os.path.join(grid_dir, '**', '*.WAV'), recursive=True))))
    if not wav_files:
        print(f"FATAL: No .wav or .WAV files found in '{grid_dir}'.")
        return None, None, None

    print(f"Found {len(wav_files)} audio files. Building grid map...")
    grid_data = defaultdict(list)
    global_min_time, global_max_time = None, None

    for file_path in wav_files:
        metadata = extract_guano_metadata_with_duration(file_path, sample_rate)
        device_id, timestamp_str, duration = metadata.get('Serial'), metadata.get('Timestamp'), metadata.get('duration_s')

        if not all([device_id, timestamp_str, duration]):
            logging.warning(f"Skipping {os.path.basename(file_path)} - Missing essential metadata.")
            continue

        try:
            original_start_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            offset_ms = mic_to_loc_id.get(device_id, {}).get('initial_offset_ms', 0)
            offset_ms += blind_sync_offsets.get(device_id, 0)
            time_correction = timedelta(milliseconds=offset_ms)
            start_time = original_start_time - time_correction
            end_time = start_time + timedelta(seconds=duration)
        except ValueError:
            logging.warning(f"Skipping {os.path.basename(file_path)} - Bad timestamp '{timestamp_str}'.")
            continue

        location_id = get_location_id(device_id, original_start_time, mic_to_loc_id)
        if location_id is None:
            logging.warning(f"Could not map {os.path.basename(file_path)} (Device: {device_id}) to a location.")
            continue

        grid_data[location_id].append({'path': file_path, 'start_time': start_time, 'end_time': end_time, 'original_start_time': original_start_time})
        if global_min_time is None or start_time < global_min_time: global_min_time = start_time
        if global_max_time is None or end_time > global_max_time: global_max_time = end_time

    for loc_id in grid_data: grid_data[loc_id].sort(key=lambda x: x['start_time'])
    print(f"Initialization complete. Found {len(grid_data)} unique locations.")
    if global_min_time and global_max_time:
        print(f"Global time range (UTC): {global_min_time.strftime('%Y-%m-%d %H:%M:%S')} to {global_max_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return grid_data, global_min_time, global_max_time

def extract_chunk_clips(grid_data: dict, chunk_start: datetime, chunk_end: datetime, sample_rate: int) -> dict:
    chunk_duration_s = (chunk_end - chunk_start).total_seconds()
    num_samples = int(chunk_duration_s * sample_rate)
    location_clips = {}

    for loc_id, files in grid_data.items():
        output_waveform = torch.zeros(1, num_samples)
        overlapping_files = [f for f in files if f['start_time'] < chunk_end and f['end_time'] > chunk_start]
        for audio_file in overlapping_files:
            try:
                file_waveform, sr = torchaudio.load(audio_file['path'])
                if sr != sample_rate: file_waveform = torchaudio.functional.resample(file_waveform, sr, sample_rate)
                
                overlap_start = max(audio_file['start_time'], chunk_start)
                overlap_end = min(audio_file['end_time'], chunk_end)
                
                offset_delta = audio_file['start_time'] - audio_file['original_start_time']
                overlap_start_original = overlap_start - offset_delta
                overlap_end_original = overlap_end - offset_delta
                
                start_sample_in_file = max(0, int((overlap_start_original - audio_file['original_start_time']).total_seconds() * sample_rate))
                end_sample_in_file = min(file_waveform.shape[1], int((overlap_end_original - audio_file['original_start_time']).total_seconds() * sample_rate))
                
                if start_sample_in_file >= end_sample_in_file: continue
                    
                segment = file_waveform[:, start_sample_in_file:end_sample_in_file]
                
                start_sample_in_output = int((overlap_start - chunk_start).total_seconds() * sample_rate)
                end_sample_in_output = start_sample_in_output + segment.shape[1]
                
                if end_sample_in_output > output_waveform.shape[1]:
                    segment = segment[:, :output_waveform.shape[1] - start_sample_in_output]
                    end_sample_in_output = output_waveform.shape[1]

                output_waveform[:, start_sample_in_output:end_sample_in_output] = segment
            except Exception as e:
                logging.error(f"Error loading segment from {os.path.basename(audio_file['path'])}: {e}")
        location_clips[loc_id] = output_waveform
    return location_clips

# --- Helper functions for de-duplicating and merging detections ---
def get_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Calculates the bounding box of a boolean mask."""
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return None
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    return (x_min, y_min, x_max, y_max)

def calculate_iou(box1: tuple, box2: tuple) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min + 1)
    inter_height = max(0, inter_y_max - inter_y_min + 1)
    intersection_area = inter_width * inter_height

    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
    
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def merge_overlapping_detections(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Merges overlapping detections of the same species based on bounding box IoU.
    Instead of discarding detections, it combines them into a single detection,
    updating the mask, confidence, and center of mass.
    """
    if not detections:
        return []

    # 1. Group detections by species name.
    species_groups = defaultdict(list)
    for det in detections:
        species_groups[det['species_name']].append(det)
    
    final_detections = []
    for species, dets in species_groups.items():
        if len(dets) <= 1:
            final_detections.extend(dets)
            continue
        
        # Add a bounding box to each detection for processing.
        for det in dets:
            det['bbox'] = get_bounding_box(det['mask'])

        # 2. Iteratively merge detections within each species group.
        while True:
            merged_in_this_pass = False
            
            # Sort by confidence. When merging, the higher confidence detection absorbs the lower one.
            dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            remaining_dets = []
            consumed_indices = set()

            for i in range(len(dets)):
                if i in consumed_indices:
                    continue

                # This is the base detection that might absorb others.
                base_det = dets[i]

                for j in range(i + 1, len(dets)):
                    if j in consumed_indices:
                        continue
                    
                    candidate_det = dets[j]
                    
                    if base_det['bbox'] is None or candidate_det['bbox'] is None:
                        continue
                    
                    iou = calculate_iou(base_det['bbox'], candidate_det['bbox'])
                    
                    if iou > iou_threshold:
                        # --- Merge candidate_det into base_det ---
                        # Combine the masks using a logical OR.
                        combined_mask = np.logical_or(base_det['mask'], candidate_det['mask'])
                        
                        # Update the base detection with the new combined properties.
                        base_det['mask'] = combined_mask
                        base_det['bbox'] = get_bounding_box(combined_mask)
                        base_det['center_of_mass'] = center_of_mass(combined_mask)
                        # Confidence is already the max due to sorting.
                        
                        # Mark the candidate as consumed.
                        consumed_indices.add(j)
                        merged_in_this_pass = True
                
                # Add the (potentially updated) base detection to the list for the next pass.
                remaining_dets.append(base_det)
            
            # Update the list of detections for the next iteration.
            dets = remaining_dets
            
            # If a full pass resulted in no merges, the process is stable.
            if not merged_in_this_pass:
                break
        
        # 3. Clean up temporary 'bbox' key and add to the final list.
        for det in dets:
            if 'bbox' in det:
                del det['bbox']
            final_detections.append(det)
                
    return final_detections

def split_detections_into_discrete_components(detections: list, min_pixel_area: int = 10) -> list:
    """
    Splits composite detections into separate detections based on connected components 
    within the mask. Filters out components smaller than min_pixel_area.
    """
    if not detections:
        return []

    split_detections = []
    
    for det in detections:
        mask = det['mask']
        
        # Label connected components (default is 4-connectivity)
        labeled_mask, num_features = label(mask)
        
        # If the mask is empty or has no substantial features
        if num_features == 0:
            continue
            
        # Iterate through found components
        for i in range(1, num_features + 1):
            component_mask = (labeled_mask == i)
            
            # Filter by size
            if np.sum(component_mask) < min_pixel_area:
                continue
                
            # Create new distinct detection
            new_det = det.copy()
            new_det['mask'] = component_mask # Update to specific component mask
            
            # Recalculate Center of Mass for this specific component
            cy, cx = center_of_mass(component_mask)
            new_det['center_of_mass'] = (cy, cx)
            
            split_detections.append(new_det)
            
    return split_detections

def run_recogniser_on_clip(waveform_tensor, sample_rate, models, configs, device):
    isolator_model, iso_processor, birdnet, recogniser = models
    iso_resize, species_map, score_thresh = configs['iso_resize'], configs['species_map'], configs['score_thresh']
    limit_species = configs.get('limit_species')

    detections = []
    complex_spec = get_complex_spectrogram(waveform_tensor, configs['stft_params']).squeeze(0)
    model_input_image = spec_to_image_for_model(complex_spec, iso_resize)

    inputs = iso_processor(images=model_input_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = isolator_model(**inputs)
    prediction = iso_processor.post_process_instance_segmentation(outputs, target_sizes=[model_input_image.size[::-1]], threshold=score_thresh)[0]
    
    if 'segmentation' not in prediction or prediction['segmentation'] is None:
        return {'image': model_input_image, 'detections': []}

    predicted_mask_img = prediction['segmentation'].cpu()
    pred_instance_ids = torch.unique(predicted_mask_img)
    pred_instance_ids = pred_instance_ids[pred_instance_ids != -1]
    if not pred_instance_ids.tolist():
        return {'image': model_input_image, 'detections': []}

    audio_duration_s = waveform_tensor.shape[1] / sample_rate
    waveform_np = waveform_tensor.squeeze().cpu().numpy()
    
    for mask_id in pred_instance_ids:
        pred_mask_log = (predicted_mask_img == mask_id)
        pred_mask_log_np = pred_mask_log.numpy()
        
        cy, cx = center_of_mass(pred_mask_log_np)
        center_time_s = (cx / pred_mask_log_np.shape[1]) * audio_duration_s

        orig_emb = get_embedding_from_waveform(waveform_np, sample_rate, birdnet, center_time_s=center_time_s)
        
        pred_mask_flipped = torch.flip(pred_mask_log, dims=[0])
        linear_mask = resample_log_mask_to_linear(pred_mask_flipped, complex_spec.shape)
        upscaled_mask = (linear_mask > 0.5).float()
        
        istft = torchaudio.transforms.InverseSpectrogram(n_fft=configs['stft_params']['n_fft'], win_length=configs['stft_params']['win_length'], hop_length=configs['stft_params']['hop_length'])
        iso_wav = istft((complex_spec * upscaled_mask).unsqueeze(0)).squeeze().cpu().numpy()
        iso_emb = get_embedding_from_waveform(iso_wav, sample_rate, birdnet, center_time_s=center_time_s)

        if orig_emb is not None and iso_emb is not None:
            final_emb = torch.FloatTensor(np.concatenate([orig_emb, iso_emb])).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = recogniser(final_emb)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
            if confidence.item() >= configs['recogniser_thresh']:
                species = species_map.get(pred_idx.item(), "Unknown")

                if limit_species and species != limit_species:
                    continue

                detections.append({
                    'species_name': species, 
                    'confidence': confidence.item(), 
                    'mask': pred_mask_log_np, 
                    'mask_id': int(mask_id.item()),
                    'center_of_mass': (cy, cx)
                })
    
    if len(detections) > 1:
        detections = merge_overlapping_detections(detections, iou_threshold=0.1)
            
    detections = split_detections_into_discrete_components(detections, min_pixel_area=10)

    return {'image': model_input_image, 'detections': detections}

def add_absolute_timing(detections: list, chunk_start: datetime, chunk_duration_s: float, sample_rate: int):
    """
    Adds absolute start/end times and crops the mask to the active bounding box.
    """
    
    for det in detections:
        mask = det['mask']
        height, width = mask.shape

        # Calculate Frequency resolution (pixels per second)
        px_per_sec = width / chunk_duration_s
        det['px_per_sec'] = px_per_sec

        # Get mask boundaries on x-axis
        _, x_coords = np.where(mask)
        if len(x_coords) > 0:
            x_min, x_max = x_coords.min(), x_coords.max()
            
            # Convert pixels to seconds
            t_start_s = (x_min / width) * chunk_duration_s
            t_end_s = (x_max / width) * chunk_duration_s
            
            det['absolute_start'] = chunk_start + timedelta(seconds=t_start_s)
            det['absolute_end'] = chunk_start + timedelta(seconds=t_end_s)
            
            cx = det['center_of_mass'][1]
            t_center_s = (cx / width) * chunk_duration_s
            det['absolute_time'] = chunk_start + timedelta(seconds=t_center_s)

            # Ensure we have at least 1 pixel width
            if x_max >= x_min:
                det['mask'] = mask[:, x_min:x_max+1]
                
                # Recalculate center of mass relative to the new cropped mask
                new_cy, new_cx = center_of_mass(det['mask'])
                det['center_of_mass'] = (new_cy, new_cx)
        else:
            # Handle empty mask edge case (should rarely happen)
            det['absolute_start'] = chunk_start
            det['absolute_end'] = chunk_start
            det['absolute_time'] = chunk_start

def check_spectral_connectivity(det1: dict, det2: dict, px_per_sec: float) -> bool:
    """
    Checks if two temporally connected detections are also spectrally connected.
    1. If they overlap in time, checks for pixel intersection in the overlapping region.
    2. If they strictly abut (gap), checks if the end of det1 connects to the start of det2.
    """
    start1, end1 = det1['absolute_start'], det1['absolute_end']
    start2, end2 = det2['absolute_start'], det2['absolute_end']
    
    # Calculate temporal overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = (overlap_end - overlap_start).total_seconds()
    
    mask1 = det1['mask']
    mask2 = det2['mask']
    
    # Case A: Significant Temporal Overlap (The common case with 50% chunk overlap)
    if overlap_duration > 0:
        # Calculate offsets to the start of the overlap region
        offset1_s = max(0.0, (overlap_start - start1).total_seconds())
        offset2_s = max(0.0, (overlap_start - start2).total_seconds())
        
        idx1 = int(round(offset1_s * px_per_sec))
        idx2 = int(round(offset2_s * px_per_sec))
        
        width_px = int(round(overlap_duration * px_per_sec))
        
        # Clamp width to available mask data
        safe_width = min(width_px, mask1.shape[1] - idx1, mask2.shape[1] - idx2)
        
        if safe_width <= 0:
            return False
            
        slice1 = mask1[:, idx1:idx1+safe_width]
        slice2 = mask2[:, idx2:idx2+safe_width]
        
        # Check if they share any active pixels in the overlap
        return np.any(np.logical_and(slice1 > 0, slice2 > 0))

    # Case B: Abutting (Gap within tolerance, effectively touching)
    # Check if the last column of det1 shares frequencies with the first column of det2
    else:
        if mask1.shape[1] == 0 or mask2.shape[1] == 0: return False
        
        # Get the last valid column of det1
        col1 = mask1[:, -1] > 0
        # Get the first valid column of det2
        col2 = mask2[:, 0] > 0
        
        return np.any(np.logical_and(col1, col2))

def repair_detections_on_boundary(detections: list, boundaries: list[datetime], tolerance_s: float = 0.1) -> list:
    """
    Merges detections of the same species that are split across chunk boundaries or 
    appear as overlapping duplicates due to chunk overlap.
    """
    if not detections:
        return []

    # Sort strictly by start time
    detections.sort(key=lambda x: x['absolute_start'])
    
    # Map indices by species to process distinct species efficiently
    species_map = defaultdict(list)
    for i, det in enumerate(detections):
        species_map[det['species_name']].append(i)
        
    indices_to_remove = set()
    
    for species, indices in species_map.items():
        if len(indices) < 2: continue
        
        i = 0
        while i < len(indices) - 1:
            idx1 = indices[i]
            current_det = detections[idx1]
            px_per_sec = current_det.get('px_per_sec', 22.4)
            
            merged_occured = False
            next_ptr = i + 1
            
            # Check subsequent detections
            while next_ptr < len(indices):
                idx2 = indices[next_ptr]
                next_det = detections[idx2]
                
                # Gap Check: If next detection starts significantly after current ends, 
                # they are not connected. Stop checking this chain.
                gap = (next_det['absolute_start'] - current_det['absolute_end']).total_seconds()
                if gap > tolerance_s:
                    break
                
                # Connectivity Check:
                # We know they are temporally close/overlapping (due to gap check above).
                # Now check if they are spectrally connected (same "blob").
                is_connected = check_spectral_connectivity(current_det, next_det, px_per_sec)
                
                if is_connected:
                    # --- PERFORM MERGE ---
                    start_diff_s = (next_det['absolute_start'] - current_det['absolute_start']).total_seconds()
                    pixel_offset = int(round(start_diff_s * px_per_sec))
                    
                    mask1 = current_det['mask']
                    mask2 = next_det['mask']
                    end_px_1 = mask1.shape[1]
                    end_px_2 = pixel_offset + mask2.shape[1]
                    new_width = max(end_px_1, end_px_2)
                    new_height = mask1.shape[0]
                    
                    new_mask = np.zeros((new_height, new_width), dtype=mask1.dtype)
                    
                    # Paste mask1 (base)
                    if mask1.shape[1] > 0:
                        new_mask[:, :mask1.shape[1]] = mask1
                    
                    # Paste mask2 (overlay) - using Maximum to merge
                    if pixel_offset < 0: 
                        # Should not happen given sorted order, but safe-guard
                        paste_start_src = abs(pixel_offset)
                        paste_start_dst = 0
                    else:
                        paste_start_src = 0
                        paste_start_dst = pixel_offset

                    paste_width = min(mask2.shape[1] - paste_start_src, new_width - paste_start_dst)
                    
                    if paste_width > 0:
                        current_slice = new_mask[:, paste_start_dst : paste_start_dst + paste_width]
                        incoming_slice = mask2[:, paste_start_src : paste_start_src + paste_width]
                        
                        new_mask[:, paste_start_dst : paste_start_dst + paste_width] = np.maximum(
                            current_slice, 
                            incoming_slice
                        )
                    
                    # Update base detection (current_det)
                    current_det['mask'] = new_mask
                    current_det['absolute_end'] = max(current_det['absolute_end'], next_det['absolute_end'])
                    current_det['confidence'] = max(current_det['confidence'], next_det['confidence'])
                    
                    # Recalculate center of mass for the merged blob
                    if np.any(new_mask):
                        cy, cx = center_of_mass(new_mask)
                        current_det['center_of_mass'] = (cy, cx)
                    
                    # Flag the merged-in detection for removal
                    indices_to_remove.add(idx2)
                    merged_occured = True
                    next_ptr += 1 # Try to merge the next one into this accumulation
                else:
                    # Temporally overlapping but spectrally distinct (e.g. different frequency note)
                    # Don't merge, but continue checking subsequent dets as they might overlap the tail
                    next_ptr += 1
            
            if merged_occured:
                # We merged items. Since indices are sorted, we can continue from the 
                # last item we checked (or simple increment, but we need to skip consumed ones).
                # Simpler to just increment i, as the consumed ones are in indices_to_remove
                # and won't be processed as base.
                i += 1
            else:
                i += 1

    return [d for i, d in enumerate(detections) if i not in indices_to_remove]

def _merge_intra_location_cluster(cluster_dets: list):
    """
    Merges detections within the same location in a temporal cluster.
    Updates the 'best' detection and marks others with '_to_remove'.
    Returns the list of valid (merged) detections for this cluster.
    """
    by_loc = defaultdict(list)
    for d in cluster_dets:
        by_loc[d['loc_id']].append(d)
        
    valid_dets = []
    
    for loc_id, dets in by_loc.items():
        if len(dets) == 1:
            valid_dets.append(dets[0])
            continue

        # Sort by confidence to keep the best one as base
        dets.sort(key=lambda x: x['confidence'], reverse=True)
        base = dets[0]
        
        # Merge others into base
        for other in dets[1:]:
            other['_to_remove'] = True
            
            new_start = min(base['absolute_start'], other['absolute_start'])
            new_end = max(base['absolute_end'], other['absolute_end'])
            
            px_per_sec = base.get('px_per_sec', 22.4)
            
            base_offset_s = (base['absolute_start'] - new_start).total_seconds()
            base_offset_px = int(round(base_offset_s * px_per_sec))
            
            other_offset_s = (other['absolute_start'] - new_start).total_seconds()
            other_offset_px = int(round(other_offset_s * px_per_sec))
            
            mask_base = base['mask']
            mask_other = other['mask']
            
            # Ensure width handles the farthest reaching mask
            end_px_base = base_offset_px + mask_base.shape[1]
            end_px_other = other_offset_px + mask_other.shape[1]
            new_width = max(end_px_base, end_px_other)
            new_height = mask_base.shape[0]
            
            new_mask = np.zeros((new_height, new_width), dtype=mask_base.dtype)
            
            if mask_base.shape[1] > 0:
                new_mask[:, base_offset_px:base_offset_px + mask_base.shape[1]] = mask_base
            
            if mask_other.shape[1] > 0:
                current_slice = new_mask[:, other_offset_px:other_offset_px + mask_other.shape[1]]
                new_mask[:, other_offset_px:other_offset_px + mask_other.shape[1]] = np.maximum(
                    current_slice, 
                    mask_other
                )
            
            base['mask'] = new_mask
            base['absolute_start'] = new_start
            base['absolute_end'] = new_end
            base['center_of_mass'] = center_of_mass(new_mask)
        
        valid_dets.append(base)
        
    return valid_dets

def calculate_sliding_mask_ios(det1: dict, det2: dict, max_tdoa_s: float) -> float:
    """
    Calculates the maximum Intersection over Smaller (IoS) by sliding det2 relative to det1
    within the time window allowed by max_tdoa_s.
    """
    mask1 = det1['mask']
    mask2 = det2['mask']
    
    area1 = (mask1 > 0).sum()
    area2 = (mask2 > 0).sum()
    min_area = min(area1, area2)
    
    if min_area == 0: return 0.0

    px_per_sec = det1.get('px_per_sec', 22.4)
    
    # Calculate the nominal offset based on absolute timestamps
    start_diff_s = (det2['absolute_start'] - det1['absolute_start']).total_seconds()
    nominal_offset_px = int(round(start_diff_s * px_per_sec))
    
    # Calculate allowed jitter in pixels
    jitter_px = int(np.ceil(max_tdoa_s * px_per_sec))
    
    # Define the range of shifts to check (det2 relative to det1)
    min_shift = nominal_offset_px - jitter_px
    max_shift = nominal_offset_px + jitter_px
    
    w1 = mask1.shape[1]
    w2 = mask2.shape[1]
    
    max_intersection = 0.0
    
    # Slide mask2 across mask1
    # Optimization: We only check shifts where masks actually overlap
    # Overlap occurs when: shift + w2 > 0  AND  shift < w1
    effective_min = max(min_shift, -w2 + 1)
    effective_max = min(max_shift, w1 - 1)
    
    if effective_max < effective_min:
        return 0.0

    # Iterate through valid shifts
    for s in range(effective_min, effective_max + 1):
        # Calculate overlapping slices
        start_in_1 = max(0, s)
        end_in_1 = min(w1, s + w2)
        
        start_in_2 = max(0, -s)
        end_in_2 = min(w2, w1 - s)
        
        width = end_in_1 - start_in_1
        if width <= 0: continue

        slice1 = mask1[:, start_in_1 : end_in_1]
        slice2 = mask2[:, start_in_2 : end_in_2]
        
        # Compute intersection for this specific lag
        # Assuming binary masks (boolean or 0/1)
        intersection_area = np.logical_and(slice1 > 0, slice2 > 0).sum()
        
        if intersection_area > max_intersection:
            max_intersection = intersection_area

    return max_intersection / min_area

def cluster_detections_into_groups(
    detection_history: dict, 
    global_counter: int, 
    safety_threshold_time: datetime,
    mic_coords: dict,
    max_global_tdoa_s: float,
    ios_threshold: float = 0.5
) -> int:
    """
    Groups detections using an Intersection over Smaller (IoS) approach.
    Only processes detections that end before 'safety_threshold_time'.
    Uses sliding IoS to account for TDOA, calculating specific max lag per mic pair.
    """
    
    # Flatten history
    all_dets = []
    for loc_id, loc_dets in detection_history.items():
        all_dets.extend(loc_dets)

    if not all_dets: return global_counter

    # Filter for detections which *started* before the threshold (the current chunk end)
    safe_dets = [d for d in all_dets if d['absolute_start'] <= safety_threshold_time]

    # Organize by Species
    by_species = defaultdict(list)
    for d in safe_dets: 
        by_species[d['species_name']].append(d)

    for species, dets in by_species.items():
        if not dets: continue
        
        # --- Graph Component Logic (Adjacency List) ---
        # Nodes are indices in 'dets'. 
        # Edges exist if IoS(d_i, d_j) > threshold.
        
        num_dets = len(dets)
        adj = defaultdict(set)
        
        # Build Graph (Naive N^2 check is fine for typical cluster sizes < 50)
        # Optimization: Only check pairs that temporally overlap within reasonable bounds (e.g. 5s)
        for i in range(num_dets):
            for j in range(i + 1, num_dets):
                d1, d2 = dets[i], dets[j]
                
                # Fast temporal overlap check (broad bounds to allow for TDOA)
                # If gap is larger than global max_tdoa + duration, they can't possibly overlap even with slide
                max_gap_s = max_global_tdoa_s + 2.0 # generous buffer
                if (d1['absolute_start'] - d2['absolute_end']).total_seconds() > max_gap_s or \
                   (d2['absolute_start'] - d1['absolute_end']).total_seconds() > max_gap_s:
                    continue
                    
                # Sliding IoS Check
                # Calculate Pairwise TDOA limit based on physical distance
                try:
                    p1 = mic_coords[d1['loc_id']]
                    p2 = mic_coords[d2['loc_id']]
                    dist = np.linalg.norm(p1 - p2)
                    pair_max_tdoa_s = (dist / 343.0) + 0.1
                except KeyError:
                    pair_max_tdoa_s = max_global_tdoa_s

                # This checks if there is ANY lag within the specific pairwise max_tdoa where they align spectrally
                if calculate_sliding_mask_ios(d1, d2, pair_max_tdoa_s) >= ios_threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        # Find Connected Components (BFS)
        visited = set()
        clusters = []
        
        for i in range(num_dets):
            if i in visited: continue
            
            component = []
            queue = [i]
            visited.add(i)
            
            while queue:
                curr = queue.pop(0)
                component.append(dets[curr])
                
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            clusters.append(component)

        # --- Process Each Identified Cluster ---
        for cluster in clusters:
            # 1. Merge same-location detections (e.g. fragmentation at same mic)
            merged_cluster = _merge_intra_location_cluster(cluster)
            
            # 2. Check 3-location rule
            unique_locs = {d['loc_id'] for d in merged_cluster}
            
            if len(unique_locs) >= 3:
                # Valid Group -> Assign/Inherit ID
                global_counter = _assign_ids_to_cluster(merged_cluster, global_counter)
            else:
                # Invalid Group -> Mark as ungrouped (remove existing ID if present)
                for d in merged_cluster:
                    if 'group_id' in d: del d['group_id']

    # Cleanup: Remove detections marked as '_to_remove' during the merge process
    for loc_id in detection_history:
        detection_history[loc_id] = [d for d in detection_history[loc_id] if not d.get('_to_remove')]

    return global_counter

def _assign_ids_to_cluster(cluster, counter):
    """
    Helper to assign IDs.
    - If any member already has an ID (from a previous frame), adopt that ID.
    - If multiple members have different IDs (merge event), adopt the lowest ID.
    - If no members have IDs, use the counter and increment.
    """
    existing_ids = {d['group_id'] for d in cluster if 'group_id' in d}
    if existing_ids:
        final_id = min(existing_ids)
    else:
        final_id = counter
        counter += 1
    for d in cluster:
        d['group_id'] = final_id
    return counter

def visualize_recent_detections_grid(detection_history, grid_data, end_time, duration_s, sample_rate):
    """
    Visualises a 3x3 grid of spectrograms for the recent history (e.g., last 15s).
    Overlays detections with unique colours per location and labels Group IDs.
    """
    start_time = end_time - timedelta(seconds=duration_s)
    
    # 1. Gather relevant detections for this window per location
    relevant_dets = defaultdict(list)
    max_dets_in_any_loc = 0
    
    for loc_id, dets in detection_history.items():
        # Find overlapping detections
        in_window = [d for d in dets if d['absolute_end'] > start_time and d['absolute_start'] < end_time]
        # Sort by start time for consistent colouring index
        in_window.sort(key=lambda x: x['absolute_start'])
        relevant_dets[loc_id] = in_window
        if len(in_window) > max_dets_in_any_loc:
            max_dets_in_any_loc = len(in_window)
    
    # If no activity, maybe skip? (Or show empty specs). Let's show empty specs for continuity.
    colors_hex = generate_rainbow_colors(max(1, max_dets_in_any_loc))
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    fig.suptitle(f"Recent Activity (Last {duration_s}s): {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}", fontsize=14)
    
    for i in range(3):
        for j in range(3):
            loc_id = (i * 3) + j + 1
            ax = axes[i, j]
            
            # A. Get Audio & Plot Spectrogram
            files = grid_data.get(loc_id, [])
            if not files:
                ax.text(0.5, 0.5, "No Files", ha='center', va='center', color='gray')
                ax.set_axis_off()
                continue
                
            waveform = extract_event_audio(files, start_time, end_time, sample_rate)
            sig_np = waveform.squeeze().numpy()
            
            if np.sum(np.abs(sig_np)) == 0:
                ax.text(0.5, 0.5, "Silence/Gap", ha='center', va='center', color='white')
                ax.set_facecolor('black')
            else:
                sig_np += np.random.normal(0, 1e-10, sig_np.shape) # Add epsilon noise to prevent log(0) warnings
                ax.specgram(sig_np, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='gray', xextent=[0, duration_s])

            # B. Overlay Detections
            loc_dets = relevant_dets[loc_id]
            for k, det in enumerate(loc_dets):
                # Determine Colour
                # Use modulo if for some reason list length mismatch, though logic prevents it
                c_hex = colors_hex[k] if k < len(colors_hex) else '#FFFFFF'
                c_rgb = [x / 255.0 for x in hex_to_rgb(c_hex)] # Normalize for matplotlib
                
                # Timing relative to window start
                t_det_start = (det['absolute_start'] - start_time).total_seconds()
                t_det_end = (det['absolute_end'] - start_time).total_seconds()
                
                # Prepare Mask Overlay
                mask_t = torch.from_numpy(det['mask']).float()
                mask_t = torch.flip(mask_t, dims=[0])
                linear_mask = resample_log_mask_to_linear(mask_t, (513, det['mask'].shape[1]))
                lin_mask_np = linear_mask.numpy()
                
                # Create Colored Overlay Image
                overlay = np.zeros((lin_mask_np.shape[0], lin_mask_np.shape[1], 4))
                overlay[..., 0] = c_rgb[0] 
                overlay[..., 1] = c_rgb[1]
                overlay[..., 2] = c_rgb[2]
                overlay[..., 3] = lin_mask_np * 0.6 # Alpha
                
                extent = [t_det_start, t_det_end, 0, sample_rate / 2]
                ax.imshow(overlay, origin='lower', extent=extent, aspect='auto')
                
                # Label Group ID if exists (centered on blob)
                if 'group_id' in det:
                    # Calculate Centroid on the linear mask
                    try:
                        cy, cx = center_of_mass(lin_mask_np)
                        
                        # Map to plot coordinates
                        # Time X:
                        det_duration = t_det_end - t_det_start
                        x_plot = t_det_start + (cx / lin_mask_np.shape[1]) * det_duration
                        
                        # Freq Y:
                        y_plot = (cy / lin_mask_np.shape[0]) * (sample_rate / 2)
                        
                        # Offset slightly up
                        y_plot += (sample_rate / 2) * 0.05
                        
                        ax.text(
                            x_plot, y_plot, 
                            f"G{det['group_id']}", 
                            color='white', fontweight='bold', fontsize=8, 
                            ha='center', va='bottom',
                            bbox=dict(facecolor=c_hex, alpha=0.8, pad=1, edgecolor='none')
                        )
                    except Exception:
                        pass # Fallback if mask is empty/error

            ax.set_title(f"Loc {loc_id}", fontsize=10, fontweight='bold', pad=2, color='black')
            ax.set_xlim(0, duration_s)
            ax.set_ylim(0, sample_rate / 2)
            if i < 2: ax.set_xticklabels([]) 
            if j > 0: ax.set_yticklabels([]) 

    plt.tight_layout()
    plt.show()

def main():
    grid_audio_directory = 'tests/localisation-grid-experiment/taukahara/sample-9mics-5mins'
    mic_id_mapping_file = 'tests/localisation-grid-experiment/taukahara/micID_to_locationID.json'
    location_mapping_file = 'tests/localisation-grid-experiment/taukahara/sensor_locations.csv'
    recogniser_config_path = 'tests/recogniser/config-recogniser.yaml'
    inference_start_time = None # e.g. '2024-10-20T00:40:00Z'
    limit_species = None # e.g. 'homo_sapien'

    blind_offset_tolerance = 3
    blind_synchronisation_offsets_path = 'tests/localisation-grid-experiment/taukahara/optimized_offsets.json'

    # Storage for offline processing
    detection_storage_dir = 'mutable_detection_data'
    if not os.path.exists(detection_storage_dir):
        os.makedirs(detection_storage_dir)

    if not all(os.path.exists(p) for p in [grid_audio_directory, mic_id_mapping_file, recogniser_config_path]):
        print("FATAL: One or more required paths do not exist.")
        return

    with open(recogniser_config_path, 'r') as f: config = yaml.safe_load(f)
    sample_rate, chunk_duration_s, chunk_overlap_s = 48000, 10, 5
    step_duration_s = chunk_duration_s - chunk_overlap_s
    
    print("\n--- Loading All Models ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    iso_model = Mask2FormerForUniversalSegmentation.from_pretrained(os.path.join(config['isolator']['model_dir'], config['isolator']['model_path'])).to(device).eval()
    iso_proc = AutoImageProcessor.from_pretrained(os.path.join(config['isolator']['model_dir'], config['isolator']['model_path']))
    birdnet = Analyzer()
    species_map = load_species_maps(os.path.join(config['paths']['output_dir'], config['paths']['species_value_map']))
    model_cfg = config['model']
    recogniser = ClassifierHead(input_size=2048, hidden_sizes=model_cfg['hidden_sizes'], output_size=len(species_map), dropout_rate=model_cfg['dropout_rate']).to(device)
    recogniser.load_state_dict(torch.load(os.path.join(config['paths']['output_dir'], "best_recogniser_head.pth"), map_location=device))
    recogniser.eval()
    
    sorted_species = sorted(list(set(species_map.values())))
    species_colors = generate_rainbow_colors(len(sorted_species))
    species_to_color_map = {species: color for species, color in zip(sorted_species, species_colors)}
    
    models = (iso_model, iso_proc, birdnet, recogniser)
    with open(os.path.join(config['isolator']['model_dir'], 'training_params.yaml'), 'r') as f: iso_training_params = yaml.safe_load(f)
    with open(os.path.join(config['isolator']['model_dir'], 'dataset_params.yaml'), 'r') as f: iso_data_params = yaml.safe_load(f)
    with open(os.path.join(config['paths']['data_dir'], config['paths']['masks_path'], 'generation_params.yaml'), 'r') as f: rec_params = yaml.safe_load(f)
    with open(os.path.join(config['paths']['output_dir'], 'full_run_config.yaml'), 'r') as f: rec_config = yaml.safe_load(f)
    localisation_config = config['localisation']
    recogniser_configs = {
        'stft_params': rec_config['isolator_dataset_params']['output']['spec_params'], # use the same STFT params as isolator was trained on
        'iso_resize': iso_training_params['resize_size'],
        'score_thresh': config['isolator']['score_threshold'],
        'species_map': species_map,
        'limit_species': limit_species,
        'recogniser_thresh': config['recogniser']['confidence_threshold']
    }
    print("--- All models loaded successfully ---\n")

    if not os.path.exists(location_mapping_file):
        print("FATAL: Sensor location CSV not found.")
        return
    mic_coords_map, _ = load_sensor_locations(location_mapping_file)
    print("Loaded sensor coordinates.")

    # Calculate Maximum TDOA (Time Difference of Arrival) for the grid
    # This is used to allow the clustering algorithm to associate sounds 
    # that arrive at different times due to the speed of sound.
    all_coords = list(mic_coords_map.values())
    max_grid_dist = 0.0
    for i in range(len(all_coords)):
        for j in range(i + 1, len(all_coords)):
            d = np.linalg.norm(all_coords[i] - all_coords[j])
            if d > max_grid_dist:
                max_grid_dist = d
    
    # Max TDOA + small buffer for mask imperfections
    grid_max_tdoa_s = (max_grid_dist / 343.0) + 0.1 
    print(f"Grid Max Diagonal: {max_grid_dist:.1f}m | Max TDOA: {grid_max_tdoa_s:.3f}s")


    print("\n--- Starting Grid Processing with Stream Consolidation ---")
    grid_data, global_start_time, global_end_time = initialize_audio_grid(grid_audio_directory, mic_id_mapping_file, blind_synchronisation_offsets_path, sample_rate)
    if not grid_data: return

    # initialize detection history buffer
    detection_history = defaultdict(list)

    current_time = global_start_time
    if inference_start_time:
        try:
            # Parse ISO string
            start_override = datetime.fromisoformat(inference_start_time.replace('Z', '+00:00'))
            
            if start_override < global_start_time:
                print(f"Warning: inference_start_time ({start_override}) is before audio starts. Using global start.")
            elif start_override > global_end_time:
                print(f"Error: inference_start_time ({start_override}) is after audio ends. Exiting.")
                return
            else:
                print(f"Jumping to specified start time: {start_override}")
                current_time = start_override
        except ValueError as e:
            print(f"Error parsing inference_start_time: {e}. Using global start.")
    total_duration_s = (global_end_time - global_start_time).total_seconds()
    # State tracking for Groups
    # Key: Group ID, Value: Last Chunk Time it was seen/updated
    group_last_seen = {} 
    localized_groups = set()
    group_id_counter = 1

    rec_stft_params = recogniser_configs['stft_params']
    while current_time < global_end_time:
        chunk_start_time = current_time
        chunk_end_time = current_time + timedelta(seconds=chunk_duration_s)
        progress = max(0, (chunk_start_time - global_start_time).total_seconds()) / total_duration_s * 100
        print(f"Inference Chunk: {chunk_start_time.strftime('%H:%M:%S')} | Analysis Window: {(chunk_start_time - timedelta(seconds=step_duration_s)).strftime('%H:%M:%S')} ({progress:.1f}%)")

        # this gives us n_locations tensors of shape (1, num_samples)
        location_clips = extract_chunk_clips(grid_data, chunk_start_time, chunk_end_time, sample_rate)

        current_chunk_images = {} # Temporary store for this loop's images
        current_raw_detections = defaultdict(list) # Temp store for debug

        for loc_id, clip_tensor in location_clips.items():
            if clip_tensor.abs().sum() == 0:
                current_chunk_images[loc_id] = Image.new('RGB', (224, 224), color='grey')
                continue
                
            # Run inference
            result = run_recogniser_on_clip(clip_tensor, sample_rate, models, recogniser_configs, device)
            
            # Store image for the *next* iteration's visualization
            current_chunk_images[loc_id] = result['image']
            
            # Process detections
            new_detections = result['detections']
            if new_detections:
                # Add location ID and Absolute Time immediately
                for d in new_detections: 
                    d['loc_id'] = loc_id
                
                # Calculate absolute timestamps based on the current chunk start
                add_absolute_timing(new_detections, chunk_start_time, chunk_duration_s, sample_rate)
                
                # We copy the list so subsequent modifications (like consolidation) don't affect the debug view of "Raw"
                current_raw_detections[loc_id] = copy.deepcopy(new_detections)

                detection_history[loc_id].extend(new_detections)
                # detection_history[loc_id] = consolidate_location_detections(detection_history[loc_id], maximum_start_time=chunk_start_time+timedelta(seconds=step_duration_s))

                # Identify boundaries for repair: 
                # 1. Start of this chunk (connects to previous chunk)
                # 2. Middle of this chunk (connects to previous chunk's end in the overlap stream)
                repair_boundaries = [
                    chunk_start_time,
                    chunk_start_time + timedelta(seconds=step_duration_s)
                ]
                
                detection_history[loc_id] = repair_detections_on_boundary(
                    detection_history[loc_id], 
                    repair_boundaries, 
                    tolerance_s=0.01 # repair detections within 10ms of boundary
                )
        
        # 1. Update Clusters
        group_id_counter = cluster_detections_into_groups(
            detection_history, 
            group_id_counter, 
            safety_threshold_time=chunk_end_time,
            mic_coords=mic_coords_map,
            max_global_tdoa_s=grid_max_tdoa_s,
            ios_threshold=0.9
        )

        # --- Visualise Recent Grid Activity (15s window) ---
        visualize_recent_detections_grid(
            detection_history, 
            grid_data, 
            end_time=chunk_end_time, 
            duration_s=15, # 10s chunk + 5s previous overlap
            sample_rate=sample_rate
        )
        # ---------------------------------------------------

        # 2. Identify Active Groups in this chunk & Register All Groups
        current_active_groups = set()
        all_visible_groups = set()

        for loc_id, dets in detection_history.items():
            for d in dets:
                if 'group_id' in d:
                    gid = d['group_id']
                    all_visible_groups.add(gid)
                    # Check if this detection intersects with current analysis window
                    # If it does, this group is still "live"
                    if d['absolute_end'] > chunk_start_time:
                        current_active_groups.add(gid)
        
        # Update "Last Seen" registry
        # We ensure that ALL groups found (even those that are fully 'safe'/inactive) are registered.
        for gid in all_visible_groups:
            if gid in current_active_groups:
                group_last_seen[gid] = current_time
            elif gid not in group_last_seen:
                # Group discovered but already inactive (safe). 
                # Register it so the localization check (step 3) sees it.
                group_last_seen[gid] = current_time


        # 3. Check for "Stale" Groups (Ready to Localize)
        # A group is ready if it was seen previously but NOT active in the current chunk
        groups_to_localize = []
        for gid, last_time in group_last_seen.items():
            if gid not in localized_groups and gid not in current_active_groups:
                # Group has stopped receiving new data
                groups_to_localize.append(gid)

        # 4. Localize Completed Groups
        if groups_to_localize:
            # Gather all detections for these groups
            all_dets_flat = [d for sublist in detection_history.values() for d in sublist if 'group_id' in d]
            for gid in groups_to_localize:
                group_dets = [d for d in all_dets_flat if d['group_id'] == gid]
                dets_by_loc = defaultdict(list)
                for d in group_dets: dets_by_loc[d['loc_id']].append(d)

                unique_group_dets = []
                for loc_id, loc_dets in dets_by_loc.items():
                    if len(loc_dets) > 1:
                        unique_group_dets.append(merge_detections_for_localization(loc_dets))
                    else:
                        unique_group_dets.append(loc_dets[0])

                # Check mic count
                # if len(unique_group_dets) >= 3:

                    # visualize_group_debug(unique_group_dets, grid_data, sample_rate)

                    #----- uncomment to save group data for later localization -----
                    # group_data = {
                    #     'group_id': gid,
                    #     'species_name': unique_group_dets[0]['species_name'],
                    #     'detections': unique_group_dets,
                    #     'approx_timestamp': group_last_seen[gid]
                    # }
                    # filename = f"group_{gid}_{int(group_last_seen[gid].timestamp())}.pkl"
                    # filepath = os.path.join(detection_storage_dir, filename)
                    # try:
                    #     with open(filepath, 'wb') as f:
                    #         pickle.dump(group_data, f)
                    #     print(f"   -> Saved Group {gid} ({group_data['species_name']}) to {filename}")
                    # except Exception as e:
                    #     print(f"   -> Failed to save Group {gid}: {e}")
                    # ---------

                    ## uncomment to perform localization immediately ###
                    # print(f"\n--- Localizing {len(groups_to_localize)} Completed Groups ---")
                
                    # attempt_localization_on_group(
                    #     unique_group_dets,
                    #     mic_coords_map,
                    #     sample_rate,
                    #     rec_stft_params,
                    #     device,
                    #     grid_data,
                    #     blind_offset_tolerance=blind_offset_tolerance,
                    #     localisation_config=localisation_config,
                    #     plot=True
                    # )
                    ## ---------

                localized_groups.add(gid)

            # # Clean up very old history to prevent memory issues (keep last 60s)
            cleanup_thresh = current_time - timedelta(seconds=60)
            for loc_id in detection_history:
                detection_history[loc_id] = [d for d in detection_history[loc_id] if d['absolute_end'] > cleanup_thresh]

        current_time += timedelta(seconds=step_duration_s)

    print("\n\n--- Grid Processing Complete ---")

if __name__ == "__main__":
    main()