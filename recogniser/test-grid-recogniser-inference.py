# recogniser/test-grid-recogniser-inference.py
# this file takes in audio from a grid and runs the recogniser on it

import os
import glob
import json
import struct
import yaml
from datetime import datetime, timedelta
from collections import defaultdict
import colorsys
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from PIL import Image
from scipy.ndimage import center_of_mass
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer
import pandas as pd
import matplotlib.pyplot as plt

from recogniser.spectrogram_tools import spectrogram_transformed

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
    This matches the exact processing pipeline of the original script.
    """
    power_spectrogram = torch.square(torch.abs(complex_spec))
    # This initial transformation is from the original script, may include normalization/clipping
    power_spectrogram = spectrogram_transformed(power_spectrogram, set_db=-10)
    
    # This second call performs the main conversion including the vertical flip
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
    
    recording = RecordingBuffer(analyzer=birdnet_analyzer, buffer=cropped_audio, rate=sample_rate, min_conf=0.0)
    recording.extract_embeddings()
    if recording.embeddings:
        return np.mean([e['embeddings'] for e in recording.embeddings], axis=0) if len(recording.embeddings) > 1 else recording.embeddings[0]['embeddings']
    return None

# --- Visualization Functions (from reference) ---

def hex_to_rgb(hex_color):
    """Converts a hex color string to an [R, G, B] list."""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def generate_rainbow_colors(num_colors: int) -> list[str]:
    """Generates a list of vibrant, distinct hex color codes."""
    if not isinstance(num_colors, int) or num_colors <= 0:
        return []
    hex_colors = []
    saturation, lightness = 1.0, 0.5
    for i in range(num_colors):
        hue = i / num_colors
        rgb_float = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb_int = tuple(int(c * 255) for c in rgb_float)
        hex_colors.append(f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}".upper())
    return hex_colors

def visualize_grid_chunk(chunk_results: dict, chunk_start_time: datetime):
    """
    Generates and displays a 3x3 subplot grid of recogniser results for a single time chunk.
    This version is optimized to maximize screen space for the spectrograms.
    """
    # Create a figure with a slightly larger size to accommodate larger plots
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    fig.suptitle(f"Acoustic Grid Analysis for Time Chunk Starting: {chunk_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC", fontsize=14)
    
    axes = axes.flatten()

    for i in range(9):
        loc_id = i + 1
        ax = axes[i]
        ax.axis('off') # Turn off the axis borders and ticks

        # Add Location ID as text in the top-left corner of the plot
        ax.text(0.03, 0.97, f"ID: {loc_id}", color='red', fontsize=10, ha='left', va='top', 
                transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.6))

        if loc_id not in chunk_results:
            ax.text(0.5, 0.5, 'No Audio Data', ha='center', va='center', transform=ax.transAxes, bbox=dict(boxstyle='round', fc='0.9'))
            continue
        
        data = chunk_results[loc_id]
        original_image = data['image']
        all_detections = data.get('detections', [])

        # Display the base spectrogram image
        image_np = np.array(original_image.convert("RGB"))
        ax.imshow(image_np)

        if not all_detections:
            continue

        # Overlay masks and labels for each detection
        colors = generate_rainbow_colors(len(all_detections))
        for j, result in enumerate(all_detections):
            mask = result['mask']
            color_rgb = hex_to_rgb(colors[j])
            color_normalised = tuple(c / 255.0 for c in color_rgb)
            
            # Create a colored overlay for the mask
            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
            overlay[mask] = color_rgb + [153] # RGB color with 60% opacity
            ax.imshow(overlay)

            # Add the species label at the center of the mask
            cy, cx = center_of_mass(mask)
            text_label = f"{result['species_name']}\n({result['confidence']*100:.1f}%) ID:{result['mask_id']}"
            ax.text(cx, 1, text_label, color=color_normalised, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.75))

    # Adjust subplot parameters to reduce whitespace and push plots up
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.05)
    plt.show()

# --- Grid Initialization and Processing Logic ---

def initialize_audio_grid(grid_dir: str, mapping_file: str, sample_rate: int):
    """
    Initializes the audio grid by scanning files, extracting metadata,
    and applying time corrections based on the initial_offset_ms from the mapping file.
    """
    print("--- Initializing Audio Grid ---")
    try:
        with open(mapping_file, 'r') as f: mic_to_loc_id = json.load(f)
    except Exception as e:
        print(f"FATAL: Could not load or parse mapping file '{mapping_file}': {e}")
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
            print(f"Warning: Skipping {os.path.basename(file_path)} - Missing essential metadata.")
            continue

        try:
            original_start_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

            # 2. Get the initial offset for the device, defaulting to 0 if not found
            offset_ms = mic_to_loc_id.get(device_id, {}).get('initial_offset_ms', 0)
            print(f'Applying offset of {offset_ms} ms for device {device_id} on dir {os.path.dirname(file_path)}')
            time_correction = timedelta(milliseconds=offset_ms)

            start_time = original_start_time - time_correction
            end_time = start_time + timedelta(seconds=duration)

        except ValueError:
            print(f"Warning: Skipping {os.path.basename(file_path)} - Bad timestamp '{timestamp_str}'.")
            continue
        # Use the original (uncorrected) start time to determine the location,
        # as the date mapping ('1013', '1017', etc.) corresponds to the file's original timestamp.
        location_id = get_location_id(device_id, original_start_time, mic_to_loc_id)
        if location_id is None:
            print(f"Warning: Could not map {os.path.basename(file_path)} (Device: {device_id}) to a location.")
            continue

        # Store the file with its corrected start and end times
        grid_data[location_id].append({
            'path': file_path, 
            'start_time': start_time, 
            'end_time': end_time,
            'original_start_time': original_start_time    
        })

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

    log_statements = []

    for loc_id, files in grid_data.items():
        output_waveform = torch.zeros(1, num_samples)
        overlapping_files = [f for f in files if f['start_time'] < chunk_end and f['end_time'] > chunk_start]
        for audio_file in overlapping_files:

            # log
            filepath = audio_file['path']
            abs_filepath = os.path.abspath(filepath)
            clickable_link = f"\033]8;;file://{abs_filepath}\007{filepath}\033]8;;\007"
            log_statements.append(f"{loc_id}: {clickable_link}")

            try:
                file_waveform, sr = torchaudio.load(audio_file['path'])
                if sr != sample_rate: 
                    file_waveform = torchaudio.functional.resample(file_waveform, sr, sample_rate)
                
                # overlap calculations use corrected timeline,
                # but sample extraction must use original timeline
                overlap_start = max(audio_file['start_time'], chunk_start)
                overlap_end = min(audio_file['end_time'], chunk_end)
                
                # The physical audio file corresponds to the original timeline
                # So we need to convert corrected times back to original timeline for indexing
                offset_delta = audio_file['start_time'] - audio_file['original_start_time']
                
                # Convert overlap times to original timeline for file indexing
                overlap_start_original = overlap_start - offset_delta
                overlap_end_original = overlap_end - offset_delta
                
                start_sample_in_file = int((overlap_start_original - audio_file['original_start_time']).total_seconds() * sample_rate)
                end_sample_in_file = int((overlap_end_original - audio_file['original_start_time']).total_seconds() * sample_rate)
                
                # Ensure indices are valid
                start_sample_in_file = max(0, start_sample_in_file)
                end_sample_in_file = min(file_waveform.shape[1], end_sample_in_file)
                if start_sample_in_file >= end_sample_in_file:
                    print(f"Warning: No overlapping samples for file {os.path.basename(audio_file['path'])}.")
                    continue
                    
                segment = file_waveform[:, start_sample_in_file:end_sample_in_file]
                
                # Place in output buffer using corrected timeline
                start_sample_in_output = int((overlap_start - chunk_start).total_seconds() * sample_rate)
                end_sample_in_output = start_sample_in_output + segment.shape[1]
                if end_sample_in_output > output_waveform.shape[1]:
                    # Trim segment if it overruns the output buffer due to rounding
                    segment = segment[:, :output_waveform.shape[1] - start_sample_in_output]
                    end_sample_in_output = output_waveform.shape[1]

                output_waveform[:, start_sample_in_output:end_sample_in_output] = segment
            except Exception as e:
                print(f"Error loading segment from {os.path.basename(audio_file['path'])}: {e}")
        location_clips[loc_id] = output_waveform

    log_statements = sorted(log_statements, key=lambda x: int(x.split(":")[0]))
    print("\n"+"\n".join(log_statements))

    return location_clips

def run_recogniser_on_clip(waveform_tensor, sample_rate, models, configs, device):
    """Applies the full pipeline to a single 10s audio clip."""
    isolator_model, iso_processor, birdnet, recogniser = models
    iso_resize, species_map, score_thresh = configs['iso_resize'], configs['species_map'], configs['score_thresh']
    
    detections = []
    complex_spec = get_complex_spectrogram(waveform_tensor, configs['stft_params']).squeeze(0)
    
    # Use the restored image function
    model_input_image = spec_to_image_for_model(complex_spec, iso_resize)

    inputs = iso_processor(images=model_input_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = isolator_model(**inputs)
    prediction = iso_processor.post_process_instance_segmentation(outputs, target_sizes=[model_input_image.size[::-1]], threshold=score_thresh)[0]
    
    if 'segmentation' not in prediction or prediction['segmentation'] is None:
        return {'image': model_input_image, 'detections': []}

    predicted_masks_log_scale = prediction['segmentation'].cpu()
    pred_instance_ids = torch.unique(predicted_masks_log_scale)
    pred_instance_ids = pred_instance_ids[pred_instance_ids != -1]
    if not pred_instance_ids.tolist():
        return {'image': model_input_image, 'detections': []}

    audio_duration_s = waveform_tensor.shape[1] / sample_rate
    waveform_np = waveform_tensor.squeeze().cpu().numpy()
    
    for mask_id in pred_instance_ids:
        # This mask is from the model and corresponds to the log-scale, flipped image
        pred_mask_log = (predicted_masks_log_scale == mask_id)
        
        # We still use the original (log-scale) mask for visualization and finding the time center
        pred_mask_log_np = pred_mask_log.numpy()
        _, center_x = center_of_mass(pred_mask_log_np)
        center_time_s = (center_x / pred_mask_log_np.shape[1]) * audio_duration_s

        with open(os.devnull, 'w') as fnull: # Suppress BirdNET output
            with redirect_stdout(fnull), redirect_stderr(fnull):
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
            if confidence.item() >= 0.99:
                species = species_map.get(pred_idx.item(), "Unknown")
                # Use the original numpy mask (pred_mask_log_np) for visualization
                detections.append({'species_name': species, 'confidence': confidence.item(), 'mask': pred_mask_log_np, 'mask_id': int(mask_id.item())})
            
    return {'image': model_input_image, 'detections': detections}

def main():
    # --- 1. Configuration ---
    grid_audio_directory = 'tests/localisation-grid-experiment/taukahara/sample-9mics-5mins'
    mic_id_mapping_file = 'tests/localisation-grid-experiment/taukahara/micID_to_locationID.json'
    recogniser_config_path = 'tests/recogniser/config-recogniser.yaml'

    if not all(os.path.exists(p) for p in [grid_audio_directory, mic_id_mapping_file, recogniser_config_path]):
        print("FATAL: One or more required paths do not exist.")
        return

    with open(recogniser_config_path, 'r') as f: config = yaml.safe_load(f)
    sample_rate, chunk_duration_s, chunk_overlap_s = 48000, 10, 5
    step_duration_s = chunk_duration_s - chunk_overlap_s
    
    # --- 2. Initialize Models ---
    print("\n--- Loading All Models ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    iso_model = Mask2FormerForUniversalSegmentation.from_pretrained(os.path.join(config['isolator']['model_dir'], config['isolator']['model_path'])).to(device).eval()
    iso_proc = AutoImageProcessor.from_pretrained(os.path.join(config['isolator']['model_dir'], config['isolator']['model_path']))
    birdnet = Analyzer()
    species_map = load_species_maps(os.path.join(config['paths']['data_dir'], config['paths']['species_value_map']))
    model_cfg = config['model']
    recogniser = ClassifierHead(input_size=2048, hidden_sizes=model_cfg['hidden_sizes'], output_size=len(species_map), dropout_rate=model_cfg['dropout_rate']).to(device)
    recogniser.load_state_dict(torch.load(os.path.join(config['paths']['output_dir'], "best_recogniser_head.pth"), map_location=device))
    recogniser.eval()
    
    models = (iso_model, iso_proc, birdnet, recogniser)
    with open(os.path.join(config['isolator']['model_dir'], 'training_params.yaml'), 'r') as f: iso_params = yaml.safe_load(f)
    with open(os.path.join(config['paths']['data_dir'], config['paths']['masks_path'], 'generation_params.yaml'), 'r') as f: rec_params = yaml.safe_load(f)
    
    recogniser_configs = {
        'stft_params': rec_params['output']['spec_params'],
        'iso_resize': iso_params['resize_size'],
        'score_thresh': config['isolator']['score_threshold'],
        'species_map': species_map,
    }
    print("--- All models loaded successfully ---\n")

    # --- 3. Scan Grid and Get Time Range ---
    grid_data, global_start_time, global_end_time = initialize_audio_grid(grid_audio_directory, mic_id_mapping_file, sample_rate)
    if not grid_data: return

    # --- 4. Main Processing Loop ---
    current_time = global_start_time
    total_duration_s = (global_end_time - global_start_time).total_seconds()
    
    print("\n--- Starting Grid Processing ---")
    while current_time < global_end_time:
        chunk_start_time = current_time
        chunk_end_time = current_time + timedelta(seconds=chunk_duration_s)
        progress = max(0, (chunk_start_time - global_start_time).total_seconds()) / total_duration_s * 100
        print(f"\rProcessing chunk: {chunk_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC ({progress:.1f}%)", end="")

        location_clips = extract_chunk_clips(grid_data, chunk_start_time, chunk_end_time, sample_rate)
        chunk_visualization_data = {}
        for loc_id, clip_tensor in location_clips.items():
            chunk_visualization_data[loc_id] = run_recogniser_on_clip(clip_tensor, sample_rate, models, recogniser_configs, device)
        
        visualize_grid_chunk(chunk_visualization_data, chunk_start_time)

        current_time += timedelta(seconds=step_duration_s)
    
    print("\n\n--- Grid Processing Complete ---")

if __name__ == "__main__":
    main()