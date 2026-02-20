import os
import yaml
import shutil
import argparse
import tempfile
import soundfile as sf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from birdnetlib import Recording, RecordingBuffer
from birdnetlib.analyzer import Analyzer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import torchaudio
import matplotlib.pyplot as plt
from matplotlib import colors
import umap
from scipy.ndimage import center_of_mass
from recogniser.spectrogram_tools import spectrogram_transformed, resample_log_mask_to_linear

# --- Configuration & Setup ---

def plot_debug_visualization(base_name, output_dir, spec_image, gt_instance_mask, gt_label_mask, predictions_info):
    """
    Creates a 4-panel plot for debugging a single audio file's processing.
    - Panel 1: Original Spectrogram Image
    - Panel 2: Ground Truth Instance Masks
    - Panel 3: Ground Truth Label Masks
    - Panel 4: Predicted Masks with IoU scores annotated.
    """
    debug_plot_dir = os.path.join(output_dir, 'debug_plots')
    os.makedirs(debug_plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 18), sharex=True, sharey=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Create a consistent, transparent colormap for overlays
    # Using 'tab20' which has 20 distinct colors
    cmap_base = plt.get_cmap('tab20')
    cmap_list = [cmap_base(i) for i in range(cmap_base.N)]
    cmap_list[0] = (0, 0, 0, 0)  # Make background (value 0) transparent
    custom_cmap = colors.ListedColormap(cmap_list)

    # Panel 1: Original Spectrogram
    ax1.imshow(spec_image)
    ax1.set_title('Spectrogram (Input to Isolator)')
    ax1.axis('off')

    # Panel 2: Ground Truth Instance Masks
    ax2.imshow(spec_image)
    ax2.imshow(gt_instance_mask, cmap=custom_cmap, alpha=0.7, interpolation='none')
    ax2.set_title('GT Instance Masks (masks/)')
    ax2.axis('off')

    # Panel 3: Ground Truth Label Masks
    ax3.imshow(spec_image)
    ax3.imshow(gt_label_mask, cmap=custom_cmap, alpha=0.7, interpolation='none')
    ax3.set_title('GT Label Masks (labels/)')
    ax3.axis('off')

    # Panel 4: Predicted Masks with IoU
    ax4.imshow(spec_image)
    ax4.set_title('Predicted Masks & Best IoU')
    ax4.axis('off')

    if predictions_info:
        # Create a single array with all predicted masks for plotting, assigning each a unique ID for color
        combined_pred_mask_plot = np.zeros_like(predictions_info[0]['mask'], dtype=int)
        for i, pred_info in enumerate(predictions_info):
            # Use i+1 as the ID to avoid background (0)
            combined_pred_mask_plot[pred_info['mask']] = i + 1

            # Annotate with IoU
            iou = pred_info['iou']
            # Find center of mass to place the text
            cy, cx = center_of_mass(pred_info['mask'])
            ax4.text(cx, cy, f'{iou:.2f}', color='white', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.7))

        ax4.imshow(combined_pred_mask_plot, cmap=custom_cmap, alpha=0.7, interpolation='none')

    plt.tight_layout()
    save_path = os.path.join(debug_plot_dir, f'{base_name}_debug.png')
    plt.savefig(save_path)
    plt.close(fig)

def load_all_configs(recogniser_config_path):
    """Loads the main recogniser config and nested configs for isolator and data."""
    print("--- Loading Configurations ---")
    
    # Load main recogniser config
    with open(recogniser_config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded recogniser config from: {recogniser_config_path}")

    # Load isolator model's original training parameters
    isolator_params_path = os.path.join(config['isolator']['model_dir'], 'training_params.yaml')
    with open(isolator_params_path, 'r') as f:
        config['isolator_params'] = yaml.safe_load(f)
    print(f"Loaded isolator training params from: {isolator_params_path}")

    # Load isolator model's original dataset generation parameters
    isolator_dataset_params_path = os.path.join(config['isolator']['model_dir'], 'dataset_params.yaml')
    with open(isolator_dataset_params_path, 'r') as f:
        config['isolator_dataset_params'] = yaml.safe_load(f)
    print(f"Loaded isolator dataset params from: {isolator_dataset_params_path}")

    # Load recogniser's dataset generation parameters
    recogniser_dataset_params_path = os.path.join(config['paths']['data_dir'], config['paths']['masks_path'], 'generation_params.yaml')
    with open(recogniser_dataset_params_path, 'r') as f:
        config['recogniser_dataset_params'] = yaml.safe_load(f)
    print(f"Loaded recogniser dataset params from: {recogniser_dataset_params_path}")

    if config['model']['baseline']:
        config['paths']['output_dir'] = config['paths']['baseline_output_dir']
        print(f'training BASELINE model, saving at: {config["paths"]["baseline_output_dir"]}')

    # Setup and clear output directory
    output_dir = config['paths']['output_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Output directory created at: {output_dir}")
    
    # Save the consolidated config to the output directory for reference
    with open(os.path.join(output_dir, "full_run_config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config

def load_species_maps(filepath, save_path=None):
    """
    Loads species mapping and creates multiple maps for convenience.
    The input CSV is headerless: class_id, species_name
    
    Args:
        filepath (str): Path to the input CSV file.
        save_path (str, optional): The file path (e.g., 'output/map.csv') 
                                   where the input CSV file should be copied. 
                                   If provided, the file will be copied. Defaults to None.
    
    Returns:
    - species_to_id: {'Species Name': 1, ...}
    - id_to_species: {1: 'Species Name', ...}
    - id_to_model_idx: {1: 0, 5: 1, 12: 2, ...} (maps arbitrary class IDs to 0-based indices)
    """
    df = pd.read_csv(filepath, header=None, names=['id', 'species_name'])
    
    if save_path:
        # **Copy the input CSV file to the specified save_path (full file path)**
        try:
            # Get the directory part of the full save_path
            target_dir = os.path.dirname(save_path)
            
            # Ensure the destination directory exists (unless the path is just a filename in the current directory)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            
            # Use shutil.copy to copy the file directly to the save_path
            shutil.copy(filepath, save_path)
            print(f"Species map CSV copied to: {save_path}")
            
        except Exception as e:
            print(f"Warning: Could not save CSV to {save_path}. Error: {e}")
            
    species_to_id = pd.Series(df.id.values, index=df.species_name).to_dict()
    id_to_species = {v: k for k, v in species_to_id.items()}
    
    # Create a mapping from the file's class ID to a 0-based index for the model
    unique_ids = sorted(df.id.unique())
    id_to_model_idx = {class_id: i for i, class_id in enumerate(unique_ids)}
    
    print(f"Loaded {len(species_to_id)} species. Model will predict {len(id_to_model_idx)} classes.")
    return species_to_id, id_to_species, id_to_model_idx


# --- Core Processing: From Audio to Embedding Pairs ---

def get_complex_spectrogram(y, stft_params):
    """Computes the complex STFT of an audio signal."""
    # Ensure y is a 2D tensor for torchaudio [channels, samples]
    if y.dim() == 1:
        y = y.unsqueeze(0)
    transform = torchaudio.transforms.Spectrogram(
        n_fft=stft_params['n_fft'],
        win_length=stft_params['win_length'], 
        hop_length=stft_params['hop_length'], 
        power=None # Returns complex tensor
    )
    return transform(y)

def spec_to_image_for_model(complex_spec, spec_params, resize_size):
    """Converts a complex spectrogram to a PIL Image formatted for the isolator model."""
    # The isolator was trained on power spectrograms processed in a specific way
    power_spectrogram = torch.square(torch.abs(complex_spec))
    power_spectrogram = spectrogram_transformed(power_spectrogram, set_db=-10)
    
    # This function should replicate the exact preprocessing used for isolator training
    image = spectrogram_transformed(
        power_spectrogram,
        to_pil=True,
        color_mode='RGB',
        log_scale=True,
        normalise='power_to_PCEN',
        resize=(resize_size, resize_size)
    )
    return image

def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union for two boolean masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0.0 # Avoid division by zero
    iou = np.sum(intersection) / np.sum(union)
    return iou

def get_embedding_from_waveform(waveform, sample_rate, birdnet_analyzer, center_time_s=None, clip_duration_s=None):
    """
    Extracts a short audio clip centered at a specific time, runs it through BirdNET,
    and returns a single embedding.
    """
    # 1. Ensure the input waveform is a CPU-based NumPy array.
    if isinstance(waveform, torch.Tensor):
        audio_buffer = waveform.cpu().numpy()
    elif isinstance(waveform, np.ndarray):
        audio_buffer = waveform
    else:
        raise TypeError(f"Unsupported waveform type: {type(waveform)}")

    # optional crop
    if center_time_s and clip_duration_s:
        # 2. Calculate the sample indices for the 3-second clip.
        center_sample = int(center_time_s * sample_rate)
        half_duration_samples = int((clip_duration_s / 2) * sample_rate)
        
        start_sample = max(0, center_sample - half_duration_samples)
        end_sample = min(len(audio_buffer), center_sample + half_duration_samples)

        # Ensure the clip is not empty
        if start_sample >= end_sample:
            raise ValueError("Calculated clip is empty. Check center_time_s and clip_duration_s.")

        cropped_audio = audio_buffer[start_sample:end_sample]
    else:
        cropped_audio = audio_buffer

    # 3. Use RecordingBuffer for in-memory processing of the cropped audio.
    recording = RecordingBuffer(
        analyzer=birdnet_analyzer,
        buffer=cropped_audio,
        rate=sample_rate,
        min_conf=0.0  # Low threshold to get an embedding for any sound.
    )
    recording.extract_embeddings()

    # 4. Return the first (and likely only) embedding found.
    if recording.embeddings:
        if len(recording.embeddings) > 1:
            embeddings_list = [e['embeddings'] for e in recording.embeddings]
            embedding = np.mean(embeddings_list, axis=0)
        else:
            embedding = recording.embeddings[0]['embeddings']
        return embedding
    else:
        return None

def get_label_from_instance_mask(label_array, instance_mask):
    """
    Finds the class ID from a label array using a boolean instance mask.
    
    Args:
        label_array (np.array): The ground truth label map where pixel values are class IDs.
        instance_mask (np.array): A boolean mask for a single instance.
        
    Returns:
        int: The class ID for that instance, or None if not found.
    """
    # Get all the label pixel values within the instance mask
    class_pixels = label_array[instance_mask]
    
    # Filter out background pixels (value 0)
    valid_pixels = class_pixels[class_pixels > 0]
    
    if len(valid_pixels) == 0:
        return None # No valid label found for this instance
        
    # Find the most common label ID (mode) in case of resize artifacts
    unique_ids, counts = np.unique(valid_pixels, return_counts=True)
    most_frequent_id = unique_ids[np.argmax(counts)]

    # subtract 1 to convert to 0-based index
    most_frequent_id -= 1
    
    return int(most_frequent_id)

def extract_embedding_pairs(config, isolator_model, isolator_processor, birdnet_analyzer, id_to_model_idx, id_to_species, device, output_dir, debug_limit=None):
    """
    Main data generation pipeline. For each audio file, it:
    1. Runs the isolator model to get predicted instance masks.
    2. Loads ground truth instance masks and parallel ground truth class label masks.
    3. Matches predicted instance masks to ground truth instance masks using IoU.
    4. If IoU > threshold, it's a positive sample, labeled with the GT species ID.
    5. If IoU <= threshold, we skip this instance
    6. For each of these samples, it creates a pair of embeddings and its corresponding label.
    """
    print("\n--- Generating Embedding Pairs from Dataset ---")
    data_dir = config['paths']['data_dir']
    audio_dir = os.path.join(data_dir, config['paths']['audio_path'])
    masks_dir = os.path.join(data_dir, config['paths']['masks_path'])
    
    all_pairs = []
    all_labels = []

    # Get params from the correct configs
    stft_params = config['recogniser_dataset_params']['output']['spec_params']
    isolator_resize_size = config['isolator_params']['resize_size']
    sample_rate = 48000
    iou_threshold = config['recogniser']['iou_threshold']

    #(TODO remove when train/val split is in a unified config) Build a lookup map for all masks, checking both train and val folders.
    all_instance_mask_paths = {}
    for subfolder in ['train/masks', 'val/masks']:
        subfolder_path = os.path.join(masks_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for f in os.listdir(subfolder_path):
                if f.endswith('.png'):
                    base_name = os.path.splitext(f)[0]
                    all_instance_mask_paths[base_name] = os.path.join(subfolder_path, f)
    all_class_label_paths = {}
    for subfolder in ['train/labels', 'val/labels']:
        subfolder_path = os.path.join(masks_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for f in os.listdir(subfolder_path):
                if f.endswith('.png'):
                    base_name = os.path.splitext(f)[0]
                    all_class_label_paths[base_name] = os.path.join(subfolder_path, f)
    if len(all_instance_mask_paths) != len(all_class_label_paths):
        print(f"Warning: Mismatch in file counts. Found {len(all_instance_mask_paths)} instance masks and {len(all_class_label_paths)} label masks.")
    print(f"Found {len(all_instance_mask_paths)} total mask/label pairs in train/ and val/ subfolders.")

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    if debug_limit:
        print(f"** DEBUG MODE: Processing only {debug_limit} files. **")
        audio_files = audio_files[:debug_limit]

    for audio_filename in tqdm(audio_files, desc="Processing Audio Files"):
        base_name = os.path.splitext(audio_filename)[0]
        if base_name not in all_instance_mask_paths or base_name not in all_class_label_paths:
            print(f"Warning: Missing mask or label for {audio_filename}. Skipping.")
            continue
        
        instance_mask_path = all_instance_mask_paths[base_name]
        label_mask_path = all_class_label_paths[base_name]
        audio_path = os.path.join(audio_dir, audio_filename)

        # 1. Load audio and ground truth masks (both instance and label)
        waveform, sr = torchaudio.load(audio_path)
        if sr != sample_rate: waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        audio_duration_s = waveform.shape[1] / sample_rate

        gt_instance_mask_array = np.array(Image.open(instance_mask_path))
        gt_label_mask_array = np.array(Image.open(label_mask_path))

        # 3. Prepare data for and run the isolator model
        complex_spec = get_complex_spectrogram(waveform, stft_params).squeeze(0)
        model_input_image = spec_to_image_for_model(complex_spec, stft_params, isolator_resize_size)
        
        inputs = isolator_processor(images=model_input_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = isolator_model(**inputs)
        
        prediction = isolator_processor.post_process_instance_segmentation(
            outputs, 
            target_sizes=[model_input_image.size[::-1]],
            threshold=config['isolator']['score_threshold']
        )[0]

        if 'segmentation' not in prediction or prediction['segmentation'] is None:
            continue
            
        predicted_masks_tensor = prediction['segmentation'].cpu()

        pred_size = predicted_masks_tensor.shape
        gt_instance_mask_tensor = torch.from_numpy(gt_instance_mask_array).float().unsqueeze(0).unsqueeze(0)
        gt_instance_mask_resized = torch.nn.functional.interpolate(gt_instance_mask_tensor, size=pred_size, mode='nearest').squeeze().numpy().astype(int)

        gt_label_mask_tensor = torch.from_numpy(gt_label_mask_array).float().unsqueeze(0).unsqueeze(0)
        gt_label_mask_resized = torch.nn.functional.interpolate(gt_label_mask_tensor, size=pred_size, mode='nearest').squeeze().numpy().astype(int)

        gt_instance_ids = np.unique(gt_instance_mask_resized)[1:]
        if len(gt_instance_ids) == 0: continue
        
        gt_instance_masks = {i: (gt_instance_mask_resized == i) for i in gt_instance_ids}
        
        pred_instance_ids = torch.unique(predicted_masks_tensor)[1:]
        pred_masks = {i.item(): (predicted_masks_tensor == i).numpy() for i in pred_instance_ids}

        predictions_info_for_plotting = []
        # First, calculate all IoUs for plotting before filtering
        for pred_id, pred_mask in pred_masks.items():
            best_iou = 0
            for gt_id, gt_mask in gt_instance_masks.items():
                iou = calculate_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
            predictions_info_for_plotting.append({'mask': pred_mask, 'iou': best_iou, 'id': pred_id})

        if debug_limit is not None:
            plot_debug_visualization(
                base_name=base_name,
                output_dir=output_dir,
                spec_image=model_input_image,
                gt_instance_mask=gt_instance_mask_resized,
                gt_label_mask=gt_label_mask_resized,
                predictions_info=predictions_info_for_plotting
            )

        # For each predicted mask, check if it's a good match (high IoU)
        for pred_id, pred_mask in pred_masks.items():
            best_iou = 0
            best_gt_instance_id = -1
            for gt_id, gt_mask in gt_instance_masks.items():
                iou = calculate_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_instance_id = gt_id
            
            label_to_assign = None
            if best_iou > iou_threshold:
                # High IoU: This is a positive sample. Find its ground truth label.
                matched_gt_instance_mask = (gt_instance_mask_resized == best_gt_instance_id)
                gt_class_id = get_label_from_instance_mask(gt_label_mask_resized, matched_gt_instance_mask)
                
                # check if species is 'unknown' (don't train classifier on unknown)
                if gt_class_id is None or id_to_species[gt_class_id] == 'unknown':
                    continue
                
                if gt_class_id is not None and gt_class_id in id_to_model_idx:
                    label_to_assign = id_to_model_idx[gt_class_id]
                        
                # If label is not found or not in our map, it will be skipped (label_to_assign remains None)
            else:
                # Low IoU: skip this instance
                continue

            # If a label was assigned, process the embedding pair.
            if label_to_assign is not None:
                # Calculate the temporal center of the mask in seconds
                _ , center_x = center_of_mass(pred_mask)
                mask_time_width = pred_mask.shape[1]
                center_time_s = (center_x / mask_time_width) * audio_duration_s

                if config['model']['baseline']:
                    center_time_s = None #no cropping for baseline

                # 2. Get original audio embedding (cropped)
                original_embedding = get_embedding_from_waveform(
                    waveform.squeeze(), sample_rate, birdnet_analyzer, center_time_s=center_time_s, clip_duration_s=3
                )
                if original_embedding is None:
                    raise ValueError(f"Could not extract original embedding for {audio_filename}. Check the audio file and BirdNET configuration.")
                
                # --- START: MASK CORRECTION AND ISOLATED AUDIO GENERATION ---
                # 1. Convert numpy mask to tensor and vertically flip it.
                pred_mask_log_tensor = torch.from_numpy(pred_mask)
                pred_mask_flipped = torch.flip(pred_mask_log_tensor, dims=[0])

                # 2. Resample the flipped mask to the linear scale of the complex spectrogram.
                linear_mask = resample_log_mask_to_linear(pred_mask_flipped, complex_spec.shape)

                # 3. Threshold to get a binary mask for multiplication.
                upscaled_pred_mask = (linear_mask > 0.5).float()

                # 4. Apply the corrected mask and perform inverse STFT.
                masked_spec = complex_spec * upscaled_pred_mask
                istft_transform = torchaudio.transforms.InverseSpectrogram(
                    n_fft=stft_params['n_fft'], win_length=stft_params['win_length'], hop_length=stft_params['hop_length']
                )
                isolated_waveform = istft_transform(masked_spec.unsqueeze(0)).squeeze()
                # --- END: MASK CORRECTION ---

                # Get the embedding for the isolated audio, centered on the same time
                isolated_embedding = get_embedding_from_waveform(
                    isolated_waveform, sample_rate, birdnet_analyzer, center_time_s=center_time_s, clip_duration_s=3
                )
                
                if isolated_embedding is not None:
                    # Create the final concatenated embedding and add it with its label to our dataset
                    if config['model']['baseline']:
                        concatenated_embedding = original_embedding
                    else:
                        concatenated_embedding = np.concatenate([original_embedding, isolated_embedding])
                    all_pairs.append(concatenated_embedding)
                    all_labels.append(label_to_assign)
                    if config['model']['baseline']:
                        break #one embedding per file
                else:
                    raise ValueError('Could not extract embedding from isolated audio. Check the audio and BirdNET configuration.')

    print(f"--- Generated {len(all_pairs)} valid embedding pairs for training. ---")
    return np.array(all_pairs), np.array(all_labels)


# --- PyTorch & Visualization Components ---

class EmbeddingPairDataset(Dataset):
    """PyTorch Dataset for our concatenated embedding pairs and labels."""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

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

    def extract_features(self, x):
        return self.feature_extractor(x)

def plot_history(history, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Epochs'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy Over Epochs'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def generate_and_save_umap_plot(model, all_embeddings, all_labels, id_to_species, model_idx_to_id, config, epoch, val_acc, device):
    print(f"  -> Generating UMAP plot for epoch {epoch+1}...")
    vis_config = config['visualization']
    output_dir = config['paths']['output_dir']
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(all_embeddings).to(device)
        hidden_activations = model.extract_features(inputs).cpu().numpy()
    
    reducer = umap.UMAP(n_neighbors=vis_config['umap_n_neighbors'], min_dist=vis_config['min_dist'], random_state=vis_config['random_state'])
    X_2d = reducer.fit_transform(hidden_activations)
    
    fig = plt.figure(figsize=(14, 12))
    
    # Map model indices back to species names for plotting
    unique_model_indices = np.unique(all_labels)
    num_species = len(unique_model_indices)
    cmap = plt.get_cmap('gist_rainbow', num_species)
    
    for i, model_idx in enumerate(unique_model_indices):
        mask = (all_labels == model_idx)
        if np.any(mask):
            if model_idx in model_idx_to_id:
                class_id = model_idx_to_id[model_idx]
                species_name = id_to_species.get(class_id, f'Unknown ID {class_id}')
            else:
                raise ValueError(f"Model index {model_idx} not found in model_idx_to_id mapping.")

            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], color=cmap(i / num_species), label=species_name, alpha=0.8, s=30)
    
    plt.title(f'UMAP of Last Hidden Layer - Epoch {epoch+1} (Val Acc: {val_acc:.4f})')
    plt.legend(title='Species', bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, f'umap_epoch_{epoch+1:03d}.png'))
    plt.close(fig)

# --- Main Training Execution ---

def main():
    parser = argparse.ArgumentParser(description="Train a recogniser head using an isolator model.")
    parser.add_argument('--config', type=str, default='recogniser/config-recogniser.yaml', help='Path to the recogniser configuration file.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a small data subset and fewer epochs.')
    args = parser.parse_args()
    # args.debug=True

    # 1. Load all configurations
    config = load_all_configs(args.config)

    # 2. Load species metadata
    species_map_path = os.path.join(config['paths']['data_dir'], 'augmented_dataset_output', config['paths']['species_value_map'])
    species_to_id, id_to_species, id_to_model_idx = load_species_maps(species_map_path, save_path=os.path.join(config['paths']['output_dir'], config['paths']['species_value_map']))
    model_idx_to_id = {v: k for k, v in id_to_model_idx.items()} # For UMAP plotting
    num_species_classes = len(id_to_model_idx)

    # 3. Initialize models (Isolator and BirdNET)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nInitializing Isolator Model (Mask2Former)...")
    model_path_full = os.path.join(config['isolator']['model_dir'], config['isolator']['model_path'])
    isolator_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path_full).to(device).eval()
    isolator_processor = AutoImageProcessor.from_pretrained(model_path_full)

    print("Initializing BirdNET Analyzer...")
    birdnet_analyzer = Analyzer()

    # 4. Generate the dataset of embedding pairs
    X_all, y_all = extract_embedding_pairs(
        config=config,
        isolator_model=isolator_model,
        isolator_processor=isolator_processor,
        birdnet_analyzer=birdnet_analyzer,
        id_to_model_idx=id_to_model_idx,
        id_to_species=id_to_species,
        device=device,
        output_dir=config['paths']['output_dir'],
        debug_limit=10 if args.debug else None
    )
    if len(X_all) == 0:
        print("\nFATAL: No valid embedding pairs could be generated. Check your data and IoU threshold. Exiting.")
        return

    # 5. Prepare data for PyTorch
    # discard classes with less than 2 samples
    unique_classes, counts = np.unique(y_all, return_counts=True)
    valid_classes = unique_classes[counts >= 2]
    mask = np.isin(y_all, valid_classes)
    print(f"Filtered out {len(y_all) - np.sum(mask)} samples with less than 2 instances per class.")
    X_all = X_all[mask]
    y_all = y_all[mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=config['training']['validation_split'],
        random_state=config['training']['random_state'],
        stratify=y_all if len(np.unique(y_all)) > 1 else None # Stratify only if possible
    )
    train_dataset = EmbeddingPairDataset(X_train, y_train)
    val_dataset = EmbeddingPairDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    print(f"\nData prepared: {len(X_train)} training pairs, {len(X_val)} validation pairs.")

    # 6. Initialize trainable classifier head
    model_cfg = config['model']
    # BirdNET embeddings are 1024-dim, we concatenate two.
    if model_cfg.get('baseline'):
        input_size = 1024
    else:
        input_size = 1024 * 2 
    num_total_classes = num_species_classes
    model = ClassifierHead(
        input_size=input_size,
        hidden_sizes=model_cfg['hidden_sizes'],
        output_size=num_total_classes,
        dropout_rate=model_cfg['dropout_rate']
    ).to(device)
    print("\nClassifier Head Architecture:"); print(model)
    print(f"Classifier configured for {num_species_classes} species = {num_total_classes} total outputs.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 7. Training Loop
    print("\n--- Starting Classifier Training ---")
    if args.debug: print(">> RUNNING IN DEBUG MODE <<\n")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    epochs = 3 if args.debug else config['training']['epochs']
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar_train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar_train.set_postfix(loss=f"{(train_loss/train_total):.4f}", acc=f"{(train_correct/train_total):.4f}")

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                pbar_val.set_postfix(loss=f"{(val_loss/val_total):.4f}", acc=f"{(val_correct/val_total):.4f}")

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss); history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc); history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} Summary | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            model_path = os.path.join(config['paths']['output_dir'], 'best_recogniser_head.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best model saved to {model_path} (Val Acc: {best_val_acc:.4f})")
            
            # Generate UMAP plot for the best model so far
            generate_and_save_umap_plot(model, X_all, y_all, id_to_species, model_idx_to_id, config, epoch, epoch_val_acc, device)

    print("\n--- Training finished. ---")
    plot_history(history, os.path.join(config['paths']['output_dir'], 'training_history.png'))
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved in: {config['paths']['output_dir']}")

if __name__ == "__main__":
    main()