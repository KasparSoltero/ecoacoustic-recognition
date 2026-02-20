import os
import yaml
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import colorsys
import math

def load_config(config_path='mask2former/config-mask2former.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def hex_to_rgb(hex_color):
    """Converts a hex color string to an [R, G, B] list."""
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def generate_rainbow_colors(num_colors: int) -> list[str]:
    """
    Generates a list of hex color codes interpolating a smooth rainbow.

    The rainbow starts at Red (hue=0.0) and ends near Violet (hue=0.75),
    maintaining full saturation and standard lightness.

    Args:
        num_colors: The number of distinct colors to generate (integer).
                    Must be non-negative.

    Returns:
        A list of hex color strings (e.g., '#FF0000').
        Returns an empty list if num_colors is 0 or negative.
        Returns ['#FF0000'] if num_colors is 1.

    Raises:
        TypeError: If num_colors is not an integer.
        ValueError: If num_colors is negative (handled by returning []).
    """
    if not isinstance(num_colors, int):
        raise TypeError("Input must be an integer.")
    if num_colors <= 0:
        return []

    hex_colors = []

    # Define HSL parameters for the rainbow
    saturation = 1.0  # Full saturation for vibrant colors
    lightness = 0.5   # Standard lightness (0.0=black, 1.0=white)
    start_hue = 0.0   # Red
    # End hue slightly before red again (e.g., 270 degrees / 360 = 0.75 for Violet)
    # Adjust this value (0.0 to 1.0) to change the end color of the rainbow
    end_hue = 0.75    # Violet

    if num_colors == 1:
        # Special case for a single color: return Red
        rgb_float = colorsys.hls_to_rgb(start_hue, lightness, saturation)
        # Convert float (0-1) to int (0-255)
        rgb_int = tuple(max(0, min(255, int(round(c * 255)))) for c in rgb_float)
        # Format as hex string and return
        return [f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}".upper()]

    # Generate colors for num_colors > 1
    for i in range(num_colors):
        # Calculate the current hue, interpolating linearly between start and end hue
        # We divide by (num_colors - 1) to ensure the last color hits end_hue exactly
        hue = start_hue + (end_hue - start_hue) * i / (num_colors - 1)

        # Convert HSL to RGB (results are floats between 0.0 and 1.0)
        rgb_float = colorsys.hls_to_rgb(hue, lightness, saturation)

        # Convert RGB floats to integer values (0-255)
        # Use round() for better accuracy and clamp values between 0 and 255
        r = max(0, min(255, int(round(rgb_float[0] * 255))))
        g = max(0, min(255, int(round(rgb_float[1] * 255))))
        b = max(0, min(255, int(round(rgb_float[2] * 255))))

        # Format the RGB tuple into a hex color string (e.g., #FF0000)
        # Use :02x formatting to ensure two digits for each component (padding with 0 if needed)
        # Use .upper() for standard uppercase hex codes
        hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
        hex_colors.append(hex_color)

    return hex_colors

def visualize_prediction(image, prediction, ground_truth_mask=None):
    """
    Visualizes the original image, predicted instances, and optionally the ground truth mask.
    - ground_truth_mask: A PIL Image of the mask, or None.
    """
    # Determine plot layout based on whether a ground truth mask is provided
    num_plots = 3 if ground_truth_mask is not None else 2
    figsize = (22, 7) if num_plots == 3 else (15, 7)
    fig, ax = plt.subplots(1, num_plots, figsize=figsize)
    
    image_np = np.array(image.convert("RGB"))
    
    # --- Plot 1: Original Image ---
    ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # --- Plot 2: Predicted Instances ---
    ax[1].imshow(image_np)
    ax[1].set_title("Predicted Vocalisation Instances")
    ax[1].axis('off')
    
    segmentation_map = prediction['segmentation'].cpu().numpy()

    if 'segments_info' in prediction and prediction['segments_info']:
        num_segments = len(prediction['segments_info'])
        rainbow_colors_hex = generate_rainbow_colors(num_segments)
        h, w, _ = image_np.shape
        mask_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        legend_handles = []

        for i, segment in enumerate(prediction['segments_info']):
            mask = (segmentation_map == segment['id'])
            hex_color = rainbow_colors_hex[i]
            rgb_color_int = hex_to_rgb(hex_color)
            mask_overlay[mask] = rgb_color_int + [153]
            
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                x_min, x_max, y_min, y_max = x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()
                color_float = np.array(rgb_color_int) / 255.0
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor=color_float, facecolor='none')
                ax[1].add_patch(rect)
                
                legend_label = f"{segment['label']} ({segment['score']:.2f}) {segment['id']}"
                legend_handles.append(patches.Patch(color=color_float, label=legend_label))
            else:
                print(f"Warning: No pixels found for segment ID {segment['id']}")
        
        ax[1].imshow(mask_overlay)
        if legend_handles:
            ax[1].legend(handles=legend_handles, loc='upper left', fontsize=8)

    # --- Plot 3: Ground Truth Mask (Conditional) ---
    if num_plots == 3:
        ax[2].set_title("Ground Truth Mask")
        ax[2].axis('off')
        
        # Convert ground truth mask to numpy array, ensure it's 2D
        gt_mask_np = np.array(ground_truth_mask)
        if gt_mask_np.ndim == 3:
            gt_mask_np = gt_mask_np[:, :, 0] # Use first channel if it's RGB

        # Find unique instance IDs, excluding background (0)
        instance_ids = np.unique(gt_mask_np)
        instance_ids = instance_ids[instance_ids != 0]

        # Create a colored version of the mask
        colored_gt_mask = np.zeros((*gt_mask_np.shape, 3), dtype=np.uint8)
        
        if len(instance_ids) > 0:
            gt_colors = generate_rainbow_colors(len(instance_ids))
            for i, instance_id in enumerate(instance_ids):
                color_rgb = hex_to_rgb(gt_colors[i])
                colored_gt_mask[gt_mask_np == instance_id] = color_rgb
        
        ax[2].imshow(colored_gt_mask)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run inference with the fine-tuned Mask2Former model.
    """
    # 1. Load Configuration
    config = load_config()
    print("Configuration loaded:")
    print(config)

    # Check config for the new visualization setting
    visualize_mask = config.get('visualize_ground_truth_mask', False)
    ground_truth_mask = None

    # 2. Define model path and check for existence
    model_path = os.path.join(config['output_dir'], "best_model")
    if not os.path.isdir(model_path):
        print(f"Error: Fine-tuned model not found at '{model_path}'.")
        print("Please run the training script first.")
        return

    # 3. Load Fine-tuned Model and Processor
    print(f"\nLoading fine-tuned model from: {model_path}...")
    # Load the specific processor saved during training
    processor = AutoImageProcessor.from_pretrained(model_path)
    # Load the fine-tuned model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval() # Set the model to evaluation mode
    print(f"Model loaded on device: {device}")

    # 4. Load and Prepare the Image
    image_path = config['test_inference_image_path']
    try:
        # We assume the test image is a spectrogram, so load as RGB
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Test image not found at '{image_path}'.")
        print("Please update 'test_inference_image_path' in the config file to point to a spectrogram.")
        return
    
    # If the setting is on, derive the mask path and try to load it
    if visualize_mask:
        try:
            # Assumes folder structure like: .../parent_folder/images/img.png and .../parent_folder/masks/img.png
            img_dir = os.path.dirname(image_path)
            parent_dir = os.path.dirname(img_dir)
            mask_filename = os.path.basename(image_path)
            mask_path = os.path.join(parent_dir, 'masks', mask_filename)
            
            ground_truth_mask = Image.open(mask_path)
            print(f"Successfully loaded ground truth mask from: {mask_path}")
        except FileNotFoundError:
            print(f"Warning: Ground truth mask not found at '{mask_path}'. Visualization will be skipped.")
            ground_truth_mask = None # Ensure it's None if not found
        except Exception as e:
            print(f"An error occurred while loading the ground truth mask: {e}")
            ground_truth_mask = None

    # The processor handles resizing and normalization based on how it was saved
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 5. Perform Inference
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. Post-process the output
    # Set a threshold to see the predictions. You might need to adjust this.
    confidence_threshold = 0.5
    print(f"\nPost-processing with confidence threshold: {confidence_threshold}")

    # We use 'post_process_instance_segmentation' as we trained for an instance task
    prediction = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
        threshold=confidence_threshold
    )[0]

    # Sort segments by score in descending order and limit to the top 5
    if 'segments_info' in prediction and prediction['segments_info']:
        all_segments = prediction['segments_info']
        sorted_segments = sorted(all_segments, key=lambda x: x['score'], reverse=True)
        top_segments = sorted_segments[:5]
        prediction['segments_info'] = top_segments # Update prediction with filtered list

    print("\nInference complete. Found the following segments (top 5 with score > 0.5):")
    if 'segments_info' in prediction and prediction['segments_info']:
        print(f'len segments_info: {len(prediction["segments_info"])}')
        for segment in prediction['segments_info']:
            # The model's config now has our custom 'vocalisation' label
            label_name = model.config.id2label[segment['label_id']]
            segment['label'] = label_name
            print(f"  - Label: {label_name}, Score: {segment['score']:.4f}, ID: {segment['id']}")
    else:
        print(f"  - No segments met the confidence threshold of {confidence_threshold}.")

    # 7. Visualize the prediction
    print("\nVisualizing results...")
    visualize_prediction(image, prediction, ground_truth_mask=ground_truth_mask)


if __name__ == "__main__":
    main()