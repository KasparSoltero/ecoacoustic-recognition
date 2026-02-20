import os
import yaml
import argparse
import ast
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import random
import ast

# Model and processing imports
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer
from scipy.ndimage import center_of_mass
import umap
from matplotlib.lines import Line2D

# Scikit-learn for metrics
from sklearn.metrics import multilabel_confusion_matrix, classification_report, average_precision_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from recogniser.spectrogram_tools import spectrogram_transformed, resample_log_mask_to_linear

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

def generate_and_save_instance_umap_plot(
    hidden_activations,
    gt_labels,
    pred_labels,
    id_to_species,
    model_idx_to_id,
    output_path
):
    """
    Generates and saves a UMAP plot of instance embeddings from the test set.
    Dot color represents the ground truth, and edge color represents the prediction.

    Args:
        hidden_activations (np.array): Array of hidden layer activations for each instance.
        gt_labels (list): List of ground truth model indices for each instance.
        pred_labels (list): List of predicted model indices for each instance.
        id_to_species (dict): Mapping from species ID to name.
        model_idx_to_id (dict): Mapping from model index to species ID.
        output_path (str): Path to save the UMAP plot.
    """
    if not hidden_activations:
        print("Warning: No instance data to generate UMAP plot. Skipping.")
        return

    print("-> Generating UMAP plot for test data instances...")
    
    # 1. Run UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer.fit_transform(np.vstack(hidden_activations))

    # 2. Prepare colors and labels for plotting
    all_model_indices = sorted(list(set(gt_labels) | set(pred_labels)))
    num_species = len(all_model_indices)
    cmap = plt.get_cmap('gist_rainbow', num_species)
    
    # Create a consistent color mapping for all species present
    color_map = {idx: cmap(i / num_species) for i, idx in enumerate(all_model_indices)}
    
    # Map each instance's GT and Pred to a color
    face_colors = [color_map[gt] for gt in gt_labels]
    # Use black for predictions of classes that might not be in the GT set (edge case)
    edge_colors = [color_map.get(pred, 'black') for pred in pred_labels]
    
    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(16, 14))
    
    ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=face_colors,
        edgecolors=edge_colors,
        linewidths=1.5,
        s=50,
        alpha=0.9
    )

    # 4. Create a custom legend
    legend_elements = []
    # Create a name-to-color map for the legend
    species_color_map = {}
    for model_idx in all_model_indices:
        species_name = id_to_species.get(model_idx_to_id.get(model_idx, -1), f"Unknown idx {model_idx}")
        species_color_map[species_name] = color_map[model_idx]
        
    # Sort by name for a clean legend
    for species_name, color in sorted(species_color_map.items()):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=species_name,
                   markerfacecolor=color, markersize=10)
        )

    # Add explanatory text to the legend
    legend_title = (
        "Species\n\n"
        "Dot Color: Ground Truth\n"
        "Edge Color: Prediction"
    )
    ax.legend(handles=legend_elements, title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    ax.set_title('UMAP of Final Hidden Layer Activations on Test Data', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"UMAP plot saved to {output_path}")

def load_species_maps(filepath):
    """
    Loads species mapping and creates multiple maps for convenience.
    Returns:
    - species_to_id: {'Species Name': 1, ...}
    - id_to_species: {1: 'Species Name', ...}
    - id_to_model_idx: {1: 0, 5: 1, 12: 2, ...} (maps arbitrary class IDs to 0-based indices)
    """
    df = pd.read_csv(filepath, header=None, names=['id', 'species_name'])
    species_to_id = pd.Series(df.id.values, index=df.species_name).to_dict()
    id_to_species = {v: k for k, v in species_to_id.items()}
    unique_ids = sorted(df.id.unique())
    id_to_model_idx = {class_id: i for i, class_id in enumerate(unique_ids)}
    print(f"Loaded {len(species_to_id)} species. Model will predict {len(id_to_model_idx)} classes.")
    return species_to_id, id_to_species, id_to_model_idx

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

def spec_to_image_for_model(complex_spec, resize_size):
    """Converts a complex spectrogram to a PIL Image formatted for the isolator model."""
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

def plot_confusion_matrix(y_true_bin, y_pred_bin, labels, output_path):
    """
    Computes and plots an aggregated confusion matrix for multi-label classification.
    This version correctly handles multi-label scenarios.
    """
    # Use sklearn's multilabel_confusion_matrix to get the components for each class
    # This returns a list of 2x2 matrices (TN, FP, FN, TP) for each class
    per_class_cm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
    
    num_labels = len(labels)
    # Create the final N x N matrix to be plotted
    final_cm = np.zeros((num_labels, num_labels), dtype=int)

    for i in range(num_labels):  # i is the TRUE class
        # Get TP for class i. This goes on the diagonal.
        # per_class_cm[i] is [[TN, FP], [FN, TP]]
        tp = per_class_cm[i][1, 1]
        final_cm[i, i] = tp

        # Now, find the misclassifications for the False Negatives of class i
        # These are instances where y_true_bin[sample, i] is 1 but y_pred_bin[sample, i] is 0
        fn_indices = np.where((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0))[0]
        
        # For each of these missed samples, see what was predicted instead
        for sample_idx in fn_indices:
            # Find what other classes were predicted for this sample
            predicted_classes = np.where(y_pred_bin[sample_idx] == 1)[0]
            if len(predicted_classes) > 0:
                # This is a substitution error. Attribute the error.
                # A simple heuristic: attribute to the first predicted class.
                # Note: In a true multi-label case, this is an approximation for visualization.
                pred_j = predicted_classes[0]
                final_cm[i, pred_j] += 1
            # If predicted_classes is empty, it was a pure "missed detection",
            # which is already accounted for in the recall score. It won't appear
            # in the N x N confusion matrix, which is standard.

    # The plot code remains the same
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    # The title should reflect what is being shown
    ax.set_title('Confusion Matrix of Detected Events\n(Rows: True Class, Cols: Predicted Class)')
    ax.set_ylabel('Ground Truth Species (for detected events)')
    ax.set_xlabel('Predicted Species')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")

def plot_roc_curves(y_true_bin, y_scores_matrix, target_names, output_path, op_point_threshold=0.5):
    """
    Computes and plots per-class and macro-average ROC curves.

    Args:
        y_true_bin (np.array): Binarized true labels (n_samples, n_classes).
        y_scores_matrix (np.array): Prediction scores (n_samples, n_classes).
        target_names (list): List of class names.
        output_path (str): Path to save the plot.
        op_point_threshold (float): The threshold for the operating point to highlight.
    """
    n_classes = len(target_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    op_points = {}

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        y_true_class = y_true_bin[:, i]
        y_scores_class = y_scores_matrix[:, i]
        
        if np.sum(y_true_class) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_class, y_scores_class)
            roc_auc[i] = auc(fpr[i], tpr[i])

            # --- NEW: Calculate the specific (FPR, TPR) for the 0.5 threshold ---
            # Get binary predictions at the operating point threshold
            y_pred_at_op_point = (y_scores_class >= op_point_threshold)
            
            # Calculate True Positives, False Positives, etc.
            tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_at_op_point, labels=[0, 1]).ravel()
            
            tpr_at_op_point = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_at_op_point = fp / (fp + tn) if (fp + tn) > 0 else 0
            op_points[i] = (fpr_at_op_point, tpr_at_op_point)
            # --- END NEW ---

        else:
            fpr[i], tpr[i], roc_auc[i] = None, None, float('nan')

    # Compute macro-average ROC curve and ROC area (same as before)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if fpr[i] is not None]))
    mean_tpr = np.zeros_like(all_fpr)
    num_valid_classes = 0
    for i in range(n_classes):
        if fpr[i] is not None:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            num_valid_classes += 1
    if num_valid_classes > 0:
        mean_tpr /= num_valid_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # --- NEW: Calculate the macro-average operating point ---
    macro_op_point_fpr = np.mean([p[0] for p in op_points.values()])
    macro_op_point_tpr = np.mean([p[1] for p in op_points.values()])
    # --- END NEW ---

    # Plotting
    plt.figure(figsize=(12, 10))
    from itertools import cycle
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    # Plot per-class ROC curves and their operating points
    op_point_label_added = False
    for i, color in zip(range(n_classes), colors):
        if roc_auc.get(i) is not None and not np.isnan(roc_auc[i]):
            # Plot the curve
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC of {target_names[i]} (area = {roc_auc[i]:.3f})")
            
            # --- NEW: Plot the operating point on the curve ---
            if i in op_points:
                point_fpr, point_tpr = op_points[i]
                label = f'Operating Point ({op_point_threshold} threshold)' if not op_point_label_added else ""
                plt.plot(point_fpr, point_tpr, 'o', color=color, markersize=8, markeredgecolor='black', label=label)
                op_point_label_added = True
            # --- END NEW ---

    # Plot macro-average ROC curve and its operating point
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-average ROC (area = {roc_auc['macro']:.3f})", color="navy", linestyle=":", linewidth=4)
    # --- NEW: Plot macro operating point ---
    plt.plot(macro_op_point_fpr, macro_op_point_tpr, 'D', color='gold', markersize=10, markeredgecolor='black', label=f'Macro Avg. at {op_point_threshold} Threshold')
    # --- END NEW ---

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) Curves with {op_point_threshold} Threshold Point")
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve plot saved to {output_path}")

# --- Main Evaluation Logic ---

def plot_visualise_isolator_predictions(
    model_input_image,
    prediction,
    filepath,
    eval_output_dir,
    ground_truth_indices,
    instance_predictions,
    file_predicted_indices,
    id_to_species,
    model_idx_to_id
):
    """
    Visualises the isolator model's output by overlaying segmentation masks on the input spectrogram.

    Saves the resulting plot to a 'visualisations' subdirectory in the evaluation output folder.

    Args:
        model_input_image (PIL.Image): The spectrogram image fed to the isolator model.
        prediction (dict): The output from the isolator's post-processing, containing segmentation masks.
        filepath (str): Path to the original audio file, used for naming the output.
        eval_output_dir (str): The main directory for saving evaluation results.
        ground_truth_indices (list): List of ground truth model indices for the file.
        instance_predictions (list): List of final predicted species per instance
        file_predicted_indices
        id_to_species (dict): Mapping from original species ID to species name.
        model_idx_to_id (dict): Mapping from 0-based model index to original species ID.
    """
    # Create a dedicated directory for these visualisations if it doesn't exist
    vis_output_dir = os.path.join(eval_output_dir, 'visualisations')
    os.makedirs(vis_output_dir, exist_ok=True)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(model_input_image)
    ax.set_axis_off()

    # Overlay predicted masks
    if 'segmentation' in prediction and prediction['segmentation'] is not None:
        segmentation_map = prediction['segmentation'].cpu()
        # Create a colormap with random colors for each instance
        instance_ids = torch.unique(segmentation_map)[1:] # Skip the background (-1)


        for instance_idx, instance_id in enumerate(instance_ids):
            # Create a boolean mask for the current instance
            mask = (segmentation_map == instance_id).numpy()
            
            # Generate a random color for the mask overlay
            color = np.random.rand(3,)
            
            # Create a colored, semi-transparent overlay
            # Shape is (height, width, 4) for RGBA
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask, :3] = color
            overlay[mask, 3] = 0.5  # 50% transparency

            ax.imshow(overlay)

            # place the text at COM
            cy, cx = center_of_mass(mask)
            # if idx not in predictions due to not being embedded . 
            if instance_id in [instance_prediction['id'] for instance_prediction in instance_predictions]:
                this_instance_prediction = [instance_prediction for instance_prediction in instance_predictions if instance_prediction['id'] == instance_id][0]
                spname = id_to_species[model_idx_to_id[this_instance_prediction['class_id']]]
                scoreinfo = this_instance_prediction['score']
                iso_scoreinfo = prediction['segments_info'][int(instance_id.item())].get('score')
                ax.text(cx, cy, 
                        spname + ', ' + f'{scoreinfo:.2f}' + f'({str(iso_scoreinfo)})', 
                        color=color, fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
            else:
                ax.text(0, 0,
                        'not embedded',
                        color=color, fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # Create an informative title
    base_filename = os.path.basename(filepath)
    try:
        gt_species_names = sorted([id_to_species[model_idx_to_id[i]] for i in ground_truth_indices])
        pred_species_names = sorted([id_to_species[model_idx_to_id[i]] for i in file_predicted_indices])
    except KeyError as e:
        print(f"Warning: A key was not found during label generation for {base_filename}. Error: {e}")
        gt_species_names = ["-error-"]
        pred_species_names = ["-error-"]


    title = (
        f"File: {base_filename}\n"
        f"Ground Truth: {', '.join(gt_species_names) or 'None'}\n"
        f"Pipeline Predictions: {', '.join(pred_species_names) or 'None'}"
    )
    ax.set_title(title, fontsize=10, wrap=True)

    # Save the figure
    output_filename = os.path.splitext(base_filename)[0] + '.png'
    output_path = os.path.join(vis_output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    # plt.show()

def evaluate(model_dir):
    """
    Main evaluation function.
    """
    print("--- Starting Recogniser Evaluation ---")
    
    # 1. Load Configurations from training run
    config_path = os.path.join(model_dir, 'full_run_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded consolidated config from training.")

    # Create output directory for this evaluation run
    eval_output_dir = os.path.join(model_dir, 'evaluation_results')
    if os.path.exists(eval_output_dir):
        shutil.rmtree(eval_output_dir)
    os.makedirs(eval_output_dir)
    print(f"Evaluation results will be saved in: {eval_output_dir}")

    # 2. Load Species Maps
    species_map_path = os.path.join(config['paths']['data_dir'], config['paths']['species_value_map'])
    species_to_id, id_to_species, id_to_model_idx = load_species_maps(species_map_path)
    model_idx_to_id = {v: k for k, v in id_to_model_idx.items()}
    num_species_classes = len(id_to_model_idx)
    
    # 3. Initialize Models
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing Isolator Model (Mask2Former)...")
    isolator_model_path = os.path.join(config['isolator']['model_dir'], config['isolator']['model_path'])
    isolator_model = Mask2FormerForUniversalSegmentation.from_pretrained(isolator_model_path).to(device).eval()
    isolator_processor = AutoImageProcessor.from_pretrained(isolator_model_path)

    print("Initializing BirdNET Analyzer...")
    birdnet_analyzer = Analyzer()

    print("Initializing Recogniser Head...")
    model_cfg = config['model']
    num_total_classes = num_species_classes
    if config['model']['baseline']:
        input_size = 1024
    else:
        input_size = 1024 * 2
    recogniser_head = ClassifierHead(
        input_size=input_size,
        hidden_sizes=model_cfg['hidden_sizes'],
        output_size=num_total_classes,
        dropout_rate=model_cfg['dropout_rate']
    ).to(device)
    recogniser_head_path = os.path.join(model_dir, 'best_recogniser_head.pth')
    recogniser_head.load_state_dict(torch.load(recogniser_head_path, map_location=device))
    recogniser_head.eval()
    print("All models loaded successfully.")

    # 4. Load and Filter Evaluation Dataset
    print("\n--- Loading and Filtering Evaluation Data from Multiple Sources ---")
    
    # Define the multiple dataset locations
    dataset_paths = [
        {
            "metadata": "/Volumes/Rectangle/NZBirdSound-OllyPowell/archive/DOC_001_Tier1/DOC_001_Tier1/001_metadata.csv",
            "audio_dir": "/Volumes/Rectangle/NZBirdSound-OllyPowell/archive/DOC_001_Tier1/DOC_001_Tier1/train_audio"
        },
        {
            "metadata": "/Volumes/Rectangle/NZBirdSound-OllyPowell/archive/DOC_002_DuncanBayParinga/DOC_002_DuncanBayParinga/002_metadata.csv",
            "audio_dir": "/Volumes/Rectangle/NZBirdSound-OllyPowell/archive/DOC_002_DuncanBayParinga/DOC_002_DuncanBayParinga/train_audio"
        },
        {
            "metadata": "/Volumes/Rectangle/NZBirdSound-OllyPowell/archive/DOC_003_XenoCanto/DOC_003_XenoCanto/003_metadata.csv",
            "audio_dir": "/Volumes/Rectangle/NZBirdSound-OllyPowell/archive/DOC_003_XenoCanto/DOC_003_XenoCanto/train_audio"
        }
    ]
    
    # Load and combine all datasets into one DataFrame
    all_dfs = []
    for d_path in dataset_paths:
        try:
            df_temp = pd.read_csv(d_path["metadata"])
            # Store the base audio path for each file, to be used later
            df_temp['audio_base_dir'] = d_path["audio_dir"] 
            all_dfs.append(df_temp)
            print(f"Loaded {len(df_temp)} records from {os.path.basename(d_path['metadata'])}")
        except FileNotFoundError:
            print(f"Warning: Metadata file not found, skipping: {d_path['metadata']}")
    
    if not all_dfs:
        print("Error: No evaluation data could be loaded. Exiting.")
        return

    df_eval_full = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined total dataset size: {len(df_eval_full)} files.")


    # --- Step 1: First, filter to samples containing ONLY classes we have trained for ---
    print("Filtering dataset to include only files with known species...")

    # Helper to combine primary and secondary labels into a single list
    def combine_labels(row):
        try:
            secondary = ast.literal_eval(row['secondary_labels']) if isinstance(row['secondary_labels'], str) and row['secondary_labels'].startswith('[') else []
            return [row['primary_label']] + secondary
        except (ValueError, SyntaxError):
            return [row['primary_label']]
    df_eval_full['all_labels_list'] = df_eval_full.apply(combine_labels, axis=1)

    # Create a boolean mask for rows where ALL labels are in our trained species map
    known_species_mask = df_eval_full['all_labels_list'].apply(
        lambda labels: all(label in species_to_id for label in labels)
    )
    df_filtered = df_eval_full[known_species_mask].copy()
    print(f"Filtered to {len(df_filtered)} files containing only known species.")

    # filter out files which were in training data
    recogniser_dataset_paths = config['recogniser_dataset_params']['paths']
    data_dir = os.path.join(config['paths']['data_dir'], recogniser_dataset_paths['dataset'])
    audio_files_to_ignore_dir = os.path.join(data_dir, recogniser_dataset_paths['vocalisations'])
    labels_format = config['recogniser_dataset_params']['input']['labels_format']
    if labels_format == 'spreadsheet':
        audio_files_to_ignore = set(os.listdir(audio_files_to_ignore_dir))
        audio_files_to_ignore = [os.path.splitext(file)[0][:-3] + '.flac' for file in audio_files_to_ignore]
    else:
        # loop folders
        audio_files_to_ignore = set()
        for subdir in os.listdir(audio_files_to_ignore_dir):
            subdirpath = os.path.join(audio_files_to_ignore_dir, subdir)
            if os.path.isdir(subdirpath):
                for file in os.listdir(subdirpath):
                    if not file.endswith('.wav'): #brittle
                        continue
                    path = os.path.join(
                        subdir,
                        os.path.splitext(file)[0][:-3] + '.flac'
                    )
                    audio_files_to_ignore.update([path])

    # remove extension and add _01.wav
    df_clean = df_filtered[~df_filtered['filename'].isin(audio_files_to_ignore)]
    print(f'removed {len(df_filtered)-len(df_clean)} files which were in the training data')

    print(f'calculating samples per class from filtered set...')
    # Count how many samples we have per class in the filtered set
    species_counts = {}
    for _, row in df_clean.iterrows():
        for label in row['all_labels_list']:
            if label in species_counts:
                species_counts[label] += 1
            else:
                species_counts[label] = 1
    print("Species counts in filtered set:")
    for species, count in species_counts.items():
        print(f"{species}: {count} samples")

    # --- Step 2: Now, limit the number of samples per class from the filtered set ---
    SAMPLES_PER_CLASS = 25
    print(f"Sampling up to {SAMPLES_PER_CLASS} files per class from the filtered set.")

    # Build an inverted index from species to file indices from the *filtered* dataframe
    class_to_file_indices = {species: [] for species in species_to_id.keys()}
    for idx, row in df_clean.iterrows():
        for label in row['all_labels_list']:
            if label in class_to_file_indices:
                class_to_file_indices[label].append(idx)

    # From the inverted index, build a set of unique file indices to include
    selected_indices = set()
    for species, indices in class_to_file_indices.items():
        random.shuffle(indices)
        selected_indices.update(indices[:SAMPLES_PER_CLASS])

    # Create the final subset DataFrame using the selected indices
    df_eval = df_clean.loc[list(selected_indices)].copy()
    print(f"Final sampled dataset size: {len(df_eval)} unique files.")


    # --- Step 3: Prepare the final list of files to process ---
    files_to_process = []
    for _, row in df_eval.iterrows():
        # All labels are known, so we can directly map them
        all_labels_text = row['all_labels_list']
        gt_class_ids = [species_to_id[label] for label in all_labels_text]
        gt_model_indices = [id_to_model_idx[cid] for cid in gt_class_ids]

        # Construct the full filepath using the 'audio_base_dir' column
        filepath = os.path.join(row['audio_base_dir'], row['filename'])

        files_to_process.append({
            'filepath': filepath,
            'ground_truth_indices': list(set(gt_model_indices)) # Use unique indices
        })

    print(f"Prepared {len(files_to_process)} files to evaluate after filtering and sampling.")

    # 5. Run Full Pipeline on Filtered Data
    print("\n--- Running End-to-End Evaluation Pipeline ---")
    stft_params = config['recogniser_dataset_params']['output']['spec_params']
    isolator_resize_size = config['isolator_params']['resize_size']
    sample_rate = 48000 # Standard sample rate
    
    all_results = []

    all_instance_hidden_activations = []
    all_instance_gt_labels = []
    all_instance_pred_labels = []

    for file_info in tqdm(files_to_process, desc="Evaluating Files"):
        filepath = file_info['filepath']
        ground_truth_indices_for_file = file_info['ground_truth_indices']
        if not os.path.exists(filepath):
            print(f"Warning: Audio file not found: {filepath}. Skipping.")
            continue

        # Load and resample audio
        waveform, sr = torchaudio.load(filepath)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        audio_duration_s = waveform.shape[1] / sample_rate

        # Get complex spectrogram and prepare for isolator
        complex_spec = get_complex_spectrogram(waveform, stft_params).squeeze(0)
        model_input_image = spec_to_image_for_model(complex_spec, isolator_resize_size)

        # Run isolator
        inputs = isolator_processor(images=model_input_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = isolator_model(**inputs)
        
        prediction = isolator_processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[model_input_image.size[::-1]],
            threshold=config['isolator']['score_threshold']
        )[0]
        
        file_max_scores = {} # Maps class_idx -> max_score
        instance_predictions = []

        file_max_scores = {}
        instance_predictions = []
        
        if 'segmentation' in prediction and prediction['segmentation'] is not None:
            predicted_masks_tensor = prediction['segmentation'].cpu()

            pred_instance_ids = torch.unique(predicted_masks_tensor)[1:]
            pred_masks = {i.item(): (predicted_masks_tensor == i).numpy() for i in pred_instance_ids}
            for instance_id, pred_mask in pred_masks.items():
                # Calculate center of mass for timing
                _, center_x = center_of_mass(pred_mask)
                center_time_s = (center_x / pred_mask.shape[1]) * audio_duration_s
                if config['model']['baseline']:
                    center_time_s = None # no croppping

                # Get original audio embedding (cropped)
                original_embedding = get_embedding_from_waveform(
                    waveform.squeeze(), sample_rate, birdnet_analyzer, center_time_s=center_time_s, clip_duration_s=3)
                if original_embedding is None:
                    print(f'instance {instance_id}, original not embedded')
                    continue
                
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
                istft = torchaudio.transforms.InverseSpectrogram(
                    n_fft=stft_params['n_fft'], win_length=stft_params['win_length'], hop_length=stft_params['hop_length']
                )
                isolated_waveform = istft(masked_spec.unsqueeze(0)).squeeze()
                # --- END: MASK CORRECTION ---
                
                isolated_embedding = get_embedding_from_waveform(
                    isolated_waveform, sample_rate, birdnet_analyzer, center_time_s=center_time_s, clip_duration_s=3)
                if isolated_embedding is None:
                    print(f'instance {instance_id}, isolated not embedded')
                    continue

                # Concatenate and classify
                if config['model']['baseline']:
                    concatenated_embedding = torch.FloatTensor(original_embedding).unsqueeze(0).to(device)
                else:
                    concatenated_embedding = torch.FloatTensor(np.concatenate([original_embedding, isolated_embedding])).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    hidden_activations = recogniser_head.extract_features(concatenated_embedding).cpu().numpy().squeeze()
                    output_logits = recogniser_head(concatenated_embedding)
                    probabilities = torch.sigmoid(output_logits).squeeze().cpu().numpy()
                    predicted_species_idx = np.argmax(probabilities)
                    if predicted_species_idx in ground_truth_indices_for_file:
                        instance_gt_label = predicted_species_idx
                    else:
                        instance_gt_label = ground_truth_indices_for_file[0] if ground_truth_indices_for_file else -1 # -1 if no GT

                    if instance_gt_label != -1:
                        all_instance_hidden_activations.append(hidden_activations)
                        all_instance_gt_labels.append(instance_gt_label)
                        all_instance_pred_labels.append(predicted_species_idx)
                
                if config['model']['baseline']:
                    for i, score in enumerate(probabilities):
                        if score > file_max_scores.get(i, 0.0):
                            file_max_scores[i] = score
                            if score > 0.5:
                                instance_predictions.append({
                                    'id': instance_id,
                                    'class_id': i,
                                    'score': score
                                })
                    break # one embedding per file

                for i, score in enumerate(probabilities):
                    if score > file_max_scores.get(i, 0.0):
                        file_max_scores[i] = score
                max_prob = probabilities[predicted_species_idx]
                if max_prob > 0.5:
                    instance_predictions.append({
                        'id': instance_id,
                        'class_id': predicted_species_idx,
                        'score': max_prob
                    })
        
        # After processing all segments, determine hard predictions for
        # the classification report based on a threshold. This is also used for visualization.
        # file_predicted_indices = [class_idx for class_idx, score in file_max_scores.items() if score > 0.5]
        file_predicted_indices = [instance_prediction['class_id'] for instance_prediction in instance_predictions]

        all_results.append({
            'filepath': filepath,
            'ground_truth': file_info['ground_truth_indices'],
            'predictions': file_predicted_indices,      # For classification_report
            'prediction_scores': file_max_scores      #For mAP and ROC AUC calculation (no score thresholds)
        })

        # This visualization part remains the same, but now uses the thresholded predictions
        plot_visualise_isolator_predictions(
            model_input_image=model_input_image,
            prediction=prediction,
            filepath=filepath,
            eval_output_dir=eval_output_dir,
            ground_truth_indices=file_info['ground_truth_indices'],
            instance_predictions=instance_predictions,
            file_predicted_indices=file_predicted_indices,
            id_to_species=id_to_species,
            model_idx_to_id=model_idx_to_id
        )   

    # 6. Calculate and Report Metrics
    print("\n--- Calculating and Reporting Metrics ---")
    y_true_all = [res['ground_truth'] for res in all_results]
    y_pred_all = [res['predictions'] for res in all_results]
    y_scores_all_dicts = [res['prediction_scores'] for res in all_results]

    # Get the class names in the correct order for reporting
    all_model_indices = list(range(num_species_classes))
    target_names = [id_to_species[model_idx_to_id[i]] for i in all_model_indices]

    # Use MultiLabelBinarizer to convert lists of labels into a binary matrix format
    binarizer = MultiLabelBinarizer(classes=all_model_indices)
    y_true_bin = binarizer.fit_transform(y_true_all)
    y_pred_bin = binarizer.transform(y_pred_all)
    
    # Generate classification report
    report = classification_report(
        y_true_bin,
        y_pred_bin,
        target_names=target_names,
        zero_division=0
    )
    print("\n--- Classification Report (at 0.5 threshold) ---")
    print(report)

    # Prepare the scores matrix needed for mAP and ROC AUC
    num_samples = len(y_scores_all_dicts)
    y_scores_matrix = np.zeros((num_samples, num_species_classes))
    for i, score_dict in enumerate(y_scores_all_dicts):
        if not score_dict:
            continue
        for class_idx, score in score_dict.items():
            if 0 <= class_idx < num_species_classes:
                y_scores_matrix[i, class_idx] = score

    # --- mAP Calculation ---
    print("\n--- Mean Average Precision (mAP) ---")
    # Calculate Average Precision for each class
    average_precisions = average_precision_score(y_true_bin, y_scores_matrix, average=None)
    mAP_score = np.nanmean(average_precisions)
    print(f"mAP (macro-average): {mAP_score:.4f}\n")

    print("\n--- Receiver Operating Characteristic (ROC) ---")

    # Calculate AUC scores
    macro_roc_auc_score = roc_auc_score(y_true_bin, y_scores_matrix, average='macro', multi_class='ovr')
    per_class_roc_auc_scores = roc_auc_score(y_true_bin, y_scores_matrix, average=None, multi_class='ovr')
    
    print(f"Macro-Averaged ROC AUC Score: {macro_roc_auc_score:.4f}\n")
    print("Per-class ROC AUC:")
    auc_results = []
    for i, class_name in enumerate(target_names):
        auc_score = per_class_roc_auc_scores[i] if not np.isnan(per_class_roc_auc_scores[i]) else 0.0
        auc_results.append({'class': class_name, 'AUC': auc_score})
        print(f"  - {class_name:<20}: {auc_score:.4f}")

    # Save ROC AUC results to a file
    roc_auc_report_path = os.path.join(eval_output_dir, 'roc_auc_report.txt')
    with open(roc_auc_report_path, 'w') as f:
        f.write("--- Receiver Operating Characteristic (ROC) Report ---\n\n")
        f.write(f"Macro-Averaged ROC AUC Score: {macro_roc_auc_score:.4f}\n\n")
        f.write("Per-class ROC AUC:\n")
        for res in sorted(auc_results, key=lambda x: x['AUC'], reverse=True):
            f.write(f"  - {res['class']:<20}: {res['AUC']:.4f}\n")
    print(f"\nROC AUC report saved to {roc_auc_report_path}")

    # Generate and save the ROC curve plot
    roc_plot_path = os.path.join(eval_output_dir, 'roc_curves.png')
    plot_roc_curves(y_true_bin, y_scores_matrix, target_names, roc_plot_path)

    # Plot Confusion Matrix (at 0.5 threshold)
    cm_path = os.path.join(eval_output_dir, 'confusion_matrix_0.5_threshold.png')
    plot_confusion_matrix(y_true_bin, y_pred_bin, target_names, cm_path)

    print("\n--- Generating UMAP Visualization ---")
    umap_plot_path = os.path.join(eval_output_dir, 'test_data_umap_plot.png')
    generate_and_save_instance_umap_plot(
        hidden_activations=all_instance_hidden_activations,
        gt_labels=all_instance_gt_labels,
        pred_labels=all_instance_pred_labels,
        id_to_species=id_to_species,
        model_idx_to_id=model_idx_to_id,
        output_path=umap_plot_path
    )

    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    model_dir = 'outputs/models-recogniser'
    evaluate(model_dir)