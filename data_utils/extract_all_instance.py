import os
import argparse
import random
from pathlib import Path
import numpy as np
import tifffile as tiff

def load_tiff_as_numpy(filepath):
    """Load a 3D TIFF file as a NumPy array."""
    return tiff.imread(filepath).astype(np.uint16)

def extract_bounding_box_3d(mask_3d, label_id, min_foreground_pixels=200):
    """
    Extract the bounding box for a given instance (label_id) in a 3D mask.
    
    Args:
        mask_3d (np.ndarray): 3D segmentation mask.
        label_id (int): Target instance label.
        min_foreground_pixels (int): Minimum required foreground pixels.
        
    Returns:
        tuple or None: (cropped_mask, bbox) if valid, where bbox is a tuple 
                       (zmin, zmax, ymin, ymax, xmin, xmax). Returns None if the instance is too small.
    """
    coords = np.where(mask_3d == label_id)
    if coords[0].size == 0:
        return None

    # Calculate bounding box coordinates
    zmin, zmax = int(coords[0].min()), int(coords[0].max())
    ymin, ymax = int(coords[1].min()), int(coords[1].max())
    xmin, xmax = int(coords[2].min()), int(coords[2].max())

    # Crop the mask using the bounding box
    cropped = mask_3d[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1].copy()

    # Convert to binary mask: set instance pixels to 1 and others to 0
    cropped[cropped != label_id] = 0
    cropped[cropped == label_id] = 1

    if np.sum(cropped) < min_foreground_pixels:
        return None

    bbox = (zmin, zmax, ymin, ymax, xmin, xmax)
    return cropped, bbox

def extract_patch_centered(volume, center, target_shape, pad_mode='constant', constant_values=0):
    """
    Extract a patch of target_shape from the volume centered at the given center.
    If the patch extends beyond the volume boundaries, pad it with constant values.
    
    Args:
        volume (np.ndarray): 3D volume to extract from.
        center (tuple): Center coordinates (z, y, x).
        target_shape (tuple): Desired patch shape (D, H, W).
        pad_mode (str): Padding mode (default is 'constant').
        constant_values (int): Value used for padding.
        
    Returns:
        np.ndarray: Extracted patch of shape target_shape.
    """
    slices = []
    pad_width = []
    for i, size in enumerate(target_shape):
        center_i = center[i]
        half = size // 2
        desired_start = center_i - half
        desired_end = desired_start + size
        vol_size = volume.shape[i]
        slice_start = max(desired_start, 0)
        slice_end = min(desired_end, vol_size)
        slices.append(slice(slice_start, slice_end))
        # Calculate required padding amounts
        pad_before = 0 if desired_start >= 0 else -desired_start
        pad_after = 0 if desired_end <= vol_size else desired_end - vol_size
        pad_width.append((pad_before, pad_after))
    patch = volume[tuple(slices)]
    if any(p != (0, 0) for p in pad_width):
        patch = np.pad(patch, pad_width, mode=pad_mode, constant_values=constant_values)
    return patch

def extract_and_save_all_instances(mask_dir, image_dir, output_mask_dir, output_image_dir, 
                                   target_shape=(32,32,32), min_foreground_pixels=300, 
                                   ignore_label_zero=True, crop_mode="random"):
    """
    Extract all instance patches from the mask and image volumes and save them to specified directories.
    
    Args:
        mask_dir (str): Directory containing mask TIFF files.
        image_dir (str): Directory containing image TIFF files.
        output_mask_dir (str): Directory to save the extracted mask patches.
        output_image_dir (str): Directory to save the extracted image patches.
        target_shape (tuple): Desired output shape (D, H, W) for the patches.
        min_foreground_pixels (int): Minimum required foreground pixels for a valid instance.
        ignore_label_zero (bool): If True, ignore label 0 (assumed background).
        crop_mode (str): Crop mode. "random" selects a random valid center, "center" uses the bounding box center.
    """
    mask_dir = Path(mask_dir)
    image_dir = Path(image_dir) if image_dir is not None else None
    output_mask_dir = Path(output_mask_dir)
    output_image_dir = Path(output_image_dir)

    # Create output directories if they don't exist
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # Process each mask file
    mask_files = sorted(mask_dir.glob("*.tif"))
    for mask_file in mask_files:
        print(f"Processing mask file: {mask_file}")
        mask_volume = load_tiff_as_numpy(mask_file)
        file_stem = mask_file.stem

        # Load the corresponding image volume if available
        image_volume = None
        if image_dir is not None:
            image_path = image_dir / mask_file.name
            if image_path.exists():
                image_volume = load_tiff_as_numpy(image_path)
            else:
                print(f"Image file not found for {mask_file.name}")

        # Get unique instance labels from the mask
        unique_ids = np.unique(mask_volume)
        if ignore_label_zero:
            unique_ids = unique_ids[unique_ids != 0]

        # Process each instance label
        for label_id in unique_ids:
            result = extract_bounding_box_3d(mask_volume, label_id, min_foreground_pixels)
            if result is None:
                print(f"Skipping instance {label_id} in {mask_file.name} due to insufficient foreground pixels")
                continue
            _, bbox = result
            # bbox: (zmin, zmax, ymin, ymax, xmin, xmax)

            # Compute the patch center based on the chosen crop_mode
            if crop_mode == "random":
                valid_center = []
                for d, s in enumerate(target_shape):
                    vol_size = mask_volume.shape[d]
                    half = s // 2
                    if d == 0:
                        bmin, bmax = bbox[0], bbox[1]
                    elif d == 1:
                        bmin, bmax = bbox[2], bbox[3]
                    elif d == 2:
                        bmin, bmax = bbox[4], bbox[5]
                    # Calculate valid center range that guarantees the patch covers [bmin, bmax]
                    center_low = max(bmax - s + half, half)
                    center_high = min(bmin + half, vol_size - s + half)
                    if center_low > center_high:
                        valid_center = None
                        break
                    valid_center.append(random.randint(center_low, center_high))
                if valid_center is None:
                    print(f"Skipping instance {label_id} in {mask_file.name} because no valid center found.")
                    continue
                center = tuple(valid_center)
            elif crop_mode == "center":
                center = []
                for d, s in enumerate(target_shape):
                    vol_size = mask_volume.shape[d]
                    half = s // 2
                    if d == 0:
                        bmin, bmax = bbox[0], bbox[1]
                    elif d == 1:
                        bmin, bmax = bbox[2], bbox[3]
                    elif d == 2:
                        bmin, bmax = bbox[4], bbox[5]
                    # Use the center of the bounding box as initial center
                    c = (bmin + bmax) // 2
                    # Compute valid range for该维度
                    center_low = max(bmax - s + half, half)
                    center_high = min(bmin + half, vol_size - s + half)
                    if center_low > center_high:
                        center = None
                        break
                    # Clamp center到合法范围内
                    c = min(max(c, center_low), center_high)
                    center.append(c)
                if center is None:
                    print(f"Skipping instance {label_id} in {mask_file.name} because no valid center found for center crop mode.")
                    continue
                center = tuple(center)
            else:
                raise ValueError(f"Unsupported crop_mode: {crop_mode}")

            # Extract a patch from the mask volume using the computed center
            patch_mask = extract_patch_centered(mask_volume, center, target_shape, pad_mode='constant', constant_values=0)
            # Binarize: set instance pixels to 1 and background to 0
            patch_mask = np.where(patch_mask == label_id, 1, 0).astype(np.uint8)

            # Save the extracted mask patch
            mask_save_path = output_mask_dir / f"{file_stem}_instance_{label_id}.tif"
            tiff.imwrite(str(mask_save_path), patch_mask)
            print(f"Saved mask patch: {mask_save_path}")

            # If an image volume is available, extract and save the corresponding patch
            if image_volume is not None:
                patch_image = extract_patch_centered(image_volume, center, target_shape, pad_mode='constant', constant_values=0)
                image_save_path = output_image_dir / f"{file_stem}_instance_{label_id}.tif"
                tiff.imwrite(str(image_save_path), patch_image)
                print(f"Saved image patch: {image_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract instance patches from TIFF files.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask TIFF files.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing image TIFF files (set to None if not available).")
    parser.add_argument("--output_mask_dir", type=str, required=True, help="Directory to save extracted mask patches.")
    parser.add_argument("--output_image_dir", type=str, required=True, help="Directory to save extracted image patches.")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[32, 32, 32], help="Desired patch size (D, H, W).")
    parser.add_argument("--min_foreground_pixels", type=int, default=20, help="Minimum required pixels for a valid instance.")
    parser.add_argument("--crop_mode", type=str, default="center", help="Crop mode: 'random' or 'center'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    # Example usage: Update the paths and parameters below as needed.
    mask_dir = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/train/masks"         # Directory containing mask TIFF files
    image_dir = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/train/images"         # Directory containing image TIFF files (set to None if not available)
    output_mask_dir = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/train/vae_instance_masks_size32"  # Directory to save extracted mask patches
    output_image_dir = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/train/vae_instance_images_size32" # Directory to save extracted image patches

    ignore_label_zero = True
    extract_and_save_all_instances(args.mask_dir, args.image_dir, args.output_mask_dir, args.output_image_dir, 
                                   args.target_shape, args.min_foreground_pixels, ignore_label_zero, args.crop_mode)
