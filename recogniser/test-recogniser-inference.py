#### maybe need to de-log the masks before applying to complex on inverse ?

import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from PIL import Image
from scipy.ndimage import center_of_mass
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import colorsys
from recogniser.spectrogram_tools import spectrogram_transformed

class ClassifierHead(nn.Module):
    """An MLP classifier head."""
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
    """
    Loads species mapping and creates multiple maps for convenience.
    Returns:
    - id_to_species: {1: 'Species Name', ...}
    - model_idx_to_species_name: {0: 'Species A', 1: 'Species B', ...}
    """
    df = pd.read_csv(filepath, header=None, names=['id', 'species_name'])
    species_to_id = pd.Series(df.id.values, index=df.species_name).to_dict()
    id_to_species = {v: k for k, v in species_to_id.items()}
    
    unique_ids = sorted(df.id.unique())
    id_to_model_idx = {class_id: i for i, class_id in enumerate(unique_ids)}
    model_idx_to_id = {v: k for k, v in id_to_model_idx.items()}
    
    model_idx_to_species_name = {
        model_idx: id_to_species[class_id]
        for model_idx, class_id in model_idx_to_id.items()
    }
    
    print(f"\nloaded {len(species_to_id)} species: {list(species_to_id.keys())}")
    return id_to_species, model_idx_to_species_name

def get_complex_spectrogram(y, stft_params):
    """Computes the complex STFT of an audio signal."""
    if y.dim() == 1:
        y = y.unsqueeze(0)
    transform = torchaudio.transforms.Spectrogram(
        n_fft=stft_params['n_fft'],
        win_length=stft_params['win_length'], 
        hop_length=stft_params['hop_length'], 
        power=None
    )
    return transform(y)

def spec_to_image_for_model(complex_spec, spec_params, resize_size):
    """Converts a complex spectrogram to a PIL Image formatted for the isolator model."""
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

def get_embedding_from_waveform(waveform, sample_rate, birdnet_analyzer, center_time_s=None, clip_duration_s=None):
    """
    Extracts a short audio clip, runs it through BirdNET, and returns an embedding.
    """
    if isinstance(waveform, torch.Tensor):
        audio_buffer = waveform.cpu().numpy()
    elif isinstance(waveform, np.ndarray):
        audio_buffer = waveform
    else:
        raise TypeError(f"Unsupported waveform type: {type(waveform)}")

    if center_time_s and clip_duration_s:
        center_sample = int(center_time_s * sample_rate)
        half_duration_samples = int((clip_duration_s / 2) * sample_rate)
        start_sample = max(0, center_sample - half_duration_samples)
        end_sample = min(len(audio_buffer), center_sample + half_duration_samples)
        if start_sample >= end_sample:
            return None
        cropped_audio = audio_buffer[start_sample:end_sample]
    else:
        cropped_audio = audio_buffer

    recording = RecordingBuffer(
        analyzer=birdnet_analyzer,
        buffer=cropped_audio,
        rate=sample_rate,
        min_conf=0.0
    )
    recording.extract_embeddings()

    if recording.embeddings:
        if len(recording.embeddings) > 1:
            return np.mean([e['embeddings'] for e in recording.embeddings], axis=0)
        return recording.embeddings[0]['embeddings']
    return None

# --- Visualization Functions  ---

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

def visualize_inference_results(original_image, all_results, output_path):
    """
    Visualizes the spectrogram with predicted masks and classification results.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    image_np = np.array(original_image.convert("RGB"))
    ax.imshow(image_np)
    ax.set_title("Recogniser Inference Results")
    ax.axis('off')

    num_segments = len(all_results)

    colors = generate_rainbow_colors(num_segments)
    
    for i, result in enumerate(all_results):
        mask = result['mask']
        species = result['species_name']
        confidence = result['confidence']
        
        # Create a colored overlay for the mask
        color_rgb = hex_to_rgb(colors[i])
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask] = color_rgb + [153]  # Add alpha for transparency
        ax.imshow(overlay)

        # Add text annotation
        cy, cx = center_of_mass(mask)
        if species == "Unclassified":
            text_label = "Unclassified"
        else:
            text_label = f"{species}\n({confidence*100:.1f}%) {result['mask_id']}"
            
        ax.text(cx, cy, text_label, color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.7))

    plt.tight_layout()
    plt.show()


# --- Main Inference Script ---

def main():
    config_path = 'tests/recogniser/config-recogniser.yaml'    
    
    # 1. Load Configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    audio_file_path = config['paths']['test_inference_audio_path']
    model_path = os.path.join(config['paths']['output_dir'], "best_recogniser_head.pth")
    
    # Load necessary nested configs
    isolator_params_path = os.path.join(config['isolator']['model_dir'], 'training_params.yaml')
    with open(isolator_params_path, 'r') as f:
        config['isolator_params'] = yaml.safe_load(f)
        
    recogniser_dataset_params_path = os.path.join(config['paths']['data_dir'], config['paths']['masks_path'], 'generation_params.yaml')
    with open(recogniser_dataset_params_path, 'r') as f:
        config['recogniser_dataset_params'] = yaml.safe_load(f)

    # 2. Load Species Metadata
    species_map_path = os.path.join(config['paths']['data_dir'], config['paths']['species_value_map'])
    _, model_idx_to_species_name = load_species_maps(species_map_path)
    num_species_classes = len(model_idx_to_species_name)

    # 3. Initialize Models
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")
    print("Initializing Isolator Model (Mask2Former)...")
    isolator_model_path = os.path.join(config['isolator']['model_dir'], config['isolator']['model_path'])
    isolator_model = Mask2FormerForUniversalSegmentation.from_pretrained(isolator_model_path).to(device).eval()
    isolator_processor = AutoImageProcessor.from_pretrained(isolator_model_path)
    print("Initializing BirdNET Analyzer...")
    birdnet_analyzer = Analyzer()
    model_cfg = config['model']
    input_size = 1024 if model_cfg.get('baseline') else 1024 * 2
    recogniser_model = ClassifierHead(
        input_size=input_size,
        hidden_sizes=model_cfg['hidden_sizes'],
        output_size=num_species_classes,
        dropout_rate=model_cfg['dropout_rate']
    ).to(device)
    recogniser_model.load_state_dict(torch.load(model_path, map_location=device))
    recogniser_model.eval()
    print("--- All models loaded successfully ---")

    # 4. Run Inference Pipeline
    print(f"--- Processing Audio File: {os.path.basename(audio_file_path)} ---")
    if not os.path.exists(audio_file_path):
        print(f"FATAL: Audio file not found at: {audio_file_path}")
        return

    waveform, sr = torchaudio.load(audio_file_path)
    sample_rate = 48000
    if sr != sample_rate:
        print(f'    sample rate: {sample_rate} Hz (resampled from {sr} Hz)')
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    else:
        print(f'    sample rate: {sample_rate} Hz')

    processing_chunk_length = config['recogniser_dataset_params']['output']['length']
    chunk_samples = processing_chunk_length * sample_rate
    total_samples = waveform.shape[1]
    num_chunks = int(np.ceil(total_samples / chunk_samples))

    print(f"    audio duration: {total_samples / sample_rate:.2f}s")
    print(f"    processing in {num_chunks} chunk(s) of {processing_chunk_length}s each")

    stft_params = config['recogniser_dataset_params']['output']['spec_params']
    print(f'    STFT params: n_fft={stft_params["n_fft"]}, win_length={stft_params["win_length"]}, hop_length={stft_params["hop_length"]}')

    isolator_resize_size = config['isolator_params']['resize_size']
    print(f'    isolator input image size: {isolator_resize_size}x{isolator_resize_size}')

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk_waveform = waveform[:, start_sample:end_sample]
        
        current_chunk_samples = chunk_waveform.shape[1] # Pad the last chunk if it's shorter than the processing length
        if current_chunk_samples < chunk_samples:
            padding_needed = chunk_samples - current_chunk_samples
            # Pad on the right (end of the audio) for the time dimension
            chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, padding_needed))
            print(f"Padded chunk {chunk_idx + 1} with {padding_needed / sample_rate:.2f}s of silence.")
        
        chunk_start_time = start_sample / sample_rate
        print(f"\n=== Processing Chunk {chunk_idx + 1}/{num_chunks} (time: {chunk_start_time:.1f}s - {end_sample/sample_rate:.1f}s) ===")

        audio_duration_s = chunk_waveform.shape[1] / sample_rate
        complex_spec = get_complex_spectrogram(chunk_waveform, stft_params).squeeze(0)
        model_input_image = spec_to_image_for_model(complex_spec, stft_params, isolator_resize_size)

        #### isolation step ####
        inputs = isolator_processor(images=model_input_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = isolator_model(**inputs)

        print(f'score threshold: {config["isolator"]["score_threshold"]}')
        prediction = isolator_processor.post_process_instance_segmentation(
            outputs, 
            target_sizes=[model_input_image.size[::-1]],
            threshold=config['isolator']['score_threshold']
        )[0]

        if 'segmentation' not in prediction or prediction['segmentation'] is None:
            print("No segments found in the audio file.")
            return

        predicted_masks = prediction['segmentation'].cpu()
        pred_instance_ids = torch.unique(predicted_masks)[1:] # Exclude background (-1)

        print(f"Found {len(pred_instance_ids)} potential vocalisation segments.")

        #### classification step ####
        all_inference_results = []

        for i, mask_id in enumerate(pred_instance_ids):
            print(f"\n--- Analyzing Segment {i+1} (Chunk {chunk_idx + 1}) ---")
            pred_mask = (predicted_masks == mask_id).numpy()
            
            _, center_x = center_of_mass(pred_mask)
            center_time_s = (center_x / pred_mask.shape[1]) * audio_duration_s
            center_s_for_embedding = None if config['model']['baseline'] else center_time_s
            
            original_embedding = get_embedding_from_waveform(
                chunk_waveform.squeeze(), sample_rate, birdnet_analyzer, 
                center_time_s=center_s_for_embedding, clip_duration_s=3
            )

            upscaled_pred_mask = torch.nn.functional.interpolate(
                torch.from_numpy(pred_mask).float().unsqueeze(0).unsqueeze(0),
                size=complex_spec.shape, mode='nearest'
            ).squeeze()
            
            masked_spec = complex_spec * upscaled_pred_mask
            istft_transform = torchaudio.transforms.InverseSpectrogram(
                n_fft=stft_params['n_fft'], win_length=stft_params['win_length'], hop_length=stft_params['hop_length']
            )
            isolated_waveform = istft_transform(masked_spec.unsqueeze(0)).squeeze()
            
            isolated_embedding = get_embedding_from_waveform(
                isolated_waveform, sample_rate, birdnet_analyzer,
                center_time_s=center_s_for_embedding, clip_duration_s=3
            )
            
            # Default result for unclassified segments
            result_data = {
                'mask': pred_mask,
                'species_name': 'Unclassified',
                'confidence': 0.0,
                'center_time_s': center_time_s,
                'mask_id': int(mask_id.item())
            }

            if original_embedding is not None and isolated_embedding is not None:
                if config['model']['baseline']:
                    final_embedding = original_embedding
                else:
                    final_embedding = np.concatenate([original_embedding, isolated_embedding])
                
                input_tensor = torch.FloatTensor(final_embedding).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = recogniser_model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predicted_idx_tensor = torch.max(probabilities, 1)

                predicted_idx = predicted_idx_tensor.item()
                predicted_species = model_idx_to_species_name.get(predicted_idx, "Unknown")
                confidence_val = confidence.item()
                
                print(f"Prediction for Segment {i+1} (centered at {center_time_s:.2f}s):")
                print(f"  -> Species: {predicted_species}")
                print(f"  -> Confidence: {confidence_val * 100:.2f}%")

                # Update result with successful classification
                result_data['species_name'] = predicted_species
                result_data['confidence'] = confidence_val
            else:
                print(f"Could not extract a valid BirdNET embedding for segment {i+1}. Skipping classification.")

            all_inference_results.append(result_data)

        # 6. Generate and save the final visualization
        output_dir = config['paths']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        audio_basename = os.path.splitext(os.path.basename(audio_file_path))[0]
        plot_path = os.path.join(output_dir, f'{audio_basename}_inference_results.png')
        
        visualize_inference_results(
            original_image=model_input_image,
            all_results=all_inference_results,
            output_path=plot_path
        )


if __name__ == "__main__":
    main()