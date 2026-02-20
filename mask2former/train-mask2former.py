import os
import yaml
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm.auto import tqdm
import albumentations as A
import glob
import matplotlib.pyplot as plt
import argparse

def load_config(config_path='mask2former/config-mask2former.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class VocalisationInstanceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform
        self.ignore_index = self.processor.ignore_index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        instance_mask = np.array(Image.open(self.mask_paths[idx]))

        if self.transform:
            transformed = self.transform(image=image, mask=instance_mask)
            image, instance_mask = transformed['image'], transformed['mask']

        instance_ids = np.unique(instance_mask)
        
        # Create a mapping from instance IDs to semantic IDs
        instance_id_to_class_id = {}
        for inst_id in instance_ids:
            inst_id = int(inst_id)
            # If the instance ID is 0, it is the background and should be ignored
            # Otherwise, map it to class ID 0 (the only class in this case)
            if inst_id == 0:
                instance_id_to_class_id[inst_id] = self.ignore_index
            else:
                instance_id_to_class_id[inst_id] = 0

        inputs = self.processor(
            [image],
            [instance_mask],
            instance_id_to_semantic_id=instance_id_to_class_id,
            return_tensors="pt"
        )
        
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        if 'class_labels' not in inputs or len(inputs["class_labels"]) == 0:
            inputs["class_labels"] = torch.tensor([], dtype=torch.long)
            inputs["mask_labels"] = torch.zeros((0, *inputs["pixel_values"].shape[1:]))

        return inputs

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
    }

def save_augmented_samples(dataloader, output_dir, prefix, num_samples=4):
    """Saves a few augmented samples from a dataloader for visual inspection."""
    plot_dir = os.path.join(output_dir, "augmented_samples")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving data sample plots to {plot_dir}")

    # Get a single batch from the dataloader
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print(f"Could not retrieve a batch from the {prefix} dataloader, skipping plot saving.")
        return

    pixel_values = batch["pixel_values"]
    mask_labels_list = batch["mask_labels"]
    
    # Don't try to plot more samples than are in the batch
    num_samples = min(num_samples, len(pixel_values))

    for i in range(num_samples):
        # Prepare the image: un-normalize and permute from (C, H, W) to (H, W, C)
        image = pixel_values[i].permute(1, 2, 0).cpu().numpy()
        # Basic un-normalization for visualization
        image_min, image_max = image.min(), image.max()
        image = (image - image_min) / (image_max - image_min) if image_max > image_min else image
        
        # Prepare the mask: combine all instance masks into one labeled image
        instance_masks = mask_labels_list[i] # Shape: (num_instances, H, W)
        combined_mask = torch.zeros(instance_masks.shape[1:], dtype=torch.int32)
        if instance_masks.shape[0] > 0: # Check if there are any instances
            for inst_idx, inst_mask in enumerate(instance_masks, 1):
                combined_mask[inst_mask.bool()] = inst_idx
        
        # Create and save the plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title(f"{prefix.capitalize()} Image Sample")
        axes[0].axis("off")
        
        axes[1].imshow(combined_mask.cpu().numpy(), cmap="viridis", vmin=0)
        axes[1].set_title(f"{prefix.capitalize()} Mask Sample")
        axes[1].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"{prefix}_sample_{i+1}.png")
        plt.savefig(save_path)
        plt.close(fig)

# --- Main Training Function ---
def main(args):
    config = load_config()
    print("Configuration for training:")
    print(config)

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data_dir = config['data_dir']
    # get generation_params.yaml and load data_augmentation_config from it
    data_config_yaml_path = os.path.join(data_dir, "generation_params.yaml")
    data_config = load_config(data_config_yaml_path)

    # save training/dataset config together to output for future reference
    with open(os.path.join(output_dir, "dataset_params.yaml"), 'w') as file:
        yaml.dump(data_config, file)
    with open(os.path.join(output_dir, "training_params.yaml"), 'w') as file:
        yaml.dump(config, file)

    resize_size = config.get('resize_size', data_config['output'].get('image_height', 512))
    print(f'resizing to: {resize_size}')

    train_image_paths = sorted(glob.glob(os.path.join(data_dir, "train/images/*.png")))
    train_mask_paths = sorted(glob.glob(os.path.join(data_dir, "train/masks/*.png")))
    val_image_paths = sorted(glob.glob(os.path.join(data_dir, "val/images/*.png")))
    val_mask_paths = sorted(glob.glob(os.path.join(data_dir, "val/masks/*.png")))

    print(f"\nFound {len(train_image_paths)} training images and {len(val_image_paths)} validation images.")

    processor = AutoImageProcessor.from_pretrained(
        config['pretrained_model_id'],
        do_resize=True,
        size={"height": resize_size, "width": resize_size},
        reduce_labels=False,
        ignore_index=255,
    )

    train_transform = A.Compose([
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)
    ])
    val_transform = None
 
    train_dataset = VocalisationInstanceDataset(train_image_paths, train_mask_paths, processor, transform=train_transform)
    val_dataset = VocalisationInstanceDataset(val_image_paths, val_mask_paths, processor, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print("\nSaving data loader samples for verification...")
    save_augmented_samples(train_dataloader, output_dir, "train", num_samples=4)
    save_augmented_samples(val_dataloader, output_dir, "val", num_samples=4)
    print("...sample plots saved.\n")
    
    id2label = {0: config['class_name']}
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        config['pretrained_model_id'],
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"\nModel and data moved to device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    # In debug mode, we only run for 1 epoch
    num_epochs = 1 if args.debug else config['epochs']
    
    if args.debug:
        print("\n" + "="*20)
        print(">> RUNNING IN DEBUG MODE <<")
        print(">> Will process only one batch for training and validation. <<")
        print("="*20 + "\n")
        # In debug mode, don't create the output directory or save models
    else:
        best_val_loss = float('inf')


    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # --- Training Phase ---
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training (Debug Mode)" if args.debug else "Training")):
            optimizer.zero_grad()
            
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = [labels.to(device) for labels in batch["mask_labels"]]
            class_labels = [labels.to(device) for labels in batch["class_labels"]]
            
            if not any(len(c) > 0 for c in class_labels):
                print("Skipping a training batch with no labels.")
                continue

            outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # In debug mode, break after the first batch
            if args.debug:
                break
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc="Validating (Debug Mode)" if args.debug else "Validating")):
                pixel_values = batch["pixel_values"].to(device)
                mask_labels = [labels.to(device) for labels in batch["mask_labels"]]
                class_labels = [labels.to(device) for labels in batch["class_labels"]]
                
                if not any(len(c) > 0 for c in class_labels):
                    print("Skipping a validation batch with no labels.")
                    continue

                outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                
                loss = outputs.loss
                total_val_loss += loss.item()
                num_val_batches += 1
                
                # In debug mode, break after the first batch
                if args.debug:
                    break
        
        # Calculate the average validation loss for the epoch
        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # --- Model saving is skipped in debug mode ---
            if not args.debug:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_save_path = os.path.join(output_dir, "best_model")
                    model.save_pretrained(model_save_path)
                    processor.save_pretrained(model_save_path)
                    print(f"New best model saved to {model_save_path} with validation loss: {best_val_loss:.4f}")
        else:
            print("No validation batches were processed; skipping model saving.")

    if args.debug:
        print("\n--- Debug run complete. No errors found. ---")
    else:
        print("\n--- Training complete! ---")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Fine-tune Mask2Former model.")
    parser.add_argument(
        "--debug",
        action="store_true", # use debug with `python train.py --debug`
        help="Run in debug mode (only one batch per epoch)."
    )
    args = parser.parse_args()
    
    main(args)