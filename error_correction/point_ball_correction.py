#!/usr/bin/env python3
"""
This script processes 3D mask TIFF files from a specified input folder. For each mask file,
it reads a corresponding CSV file (with the same base name and a .csv extension) from a given CSV folder.
The CSV file must contain two columns: 'id' and 'label'. The script identifies error IDs 
(where label equals 1) and computes the average dimensions (depth, height, width) of valid (non-error and non-background) labels.
For each error ID in the mask, the script:
  - Computes the center of the error region (or selects a random point within the region if in random mode).
  - Removes the original error region (setting it to background, assumed to be 0).
  - Inserts an ellipsoid at the computed center with dimensions equal to the average sizes,
    labeling the ellipsoid with the error ID.
The modified mask is then saved to the output folder using the same filename.
"""

import os
import argparse
import csv
import numpy as np
import tifffile

def process_mask_file(filepath, csv_folder, output_folder, placement_mode):
    """
    Process a single 3D mask TIFF file.

    For a given mask file and its corresponding CSV file (with the same base name), this function:
      - Reads the mask file.
      - Reads the CSV file to extract error IDs (where label == 1) using the csv module.
      - Computes the average dimensions (depth, height, width) for valid labels (excluding error IDs and background).
      - For each error ID:
          * Computes the center of its mask region (or randomly selects a point in the region if mode is "random").
          * Removes the original error region from the mask.
          * Inserts an ellipsoid with dimensions equal to the average sizes at the chosen center,
            labeling the ellipsoid with the error ID.
      - Saves the modified mask to the output folder with the same filename.
    """
    # Read the 3D mask data
    mask = tifffile.imread(filepath)
    
    # Determine the corresponding CSV file based on the mask filename
    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    csv_filename = base_name + '.csv'
    csv_path = os.path.join(csv_folder, csv_filename)
    
    # If the corresponding CSV exists, read error IDs (where label == 1) using csv module
    error_ids = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Assuming CSV columns 'id' and 'label'; compare label as string
                if row.get('label') == '1':
                    try:
                        error_ids.append(int(row.get('id')))
                    except (ValueError, TypeError):
                        print(f"Invalid id value in {csv_filename}: {row.get('id')}")
    else:
        print(f"CSV file {csv_filename} does not exist. Skipping processing of {filename}.")
        return

    # Get all unique IDs in the mask
    unique_ids = np.unique(mask)
    # Exclude error IDs and background (assumed background is 0)
    valid_ids = [uid for uid in unique_ids if uid not in error_ids and uid != 0]
    
    # Collect dimensions (depth, height, width) for valid labels
    sizes = []
    for uid in valid_ids:
        coords = np.array(np.where(mask == uid))
        # Calculate size for each dimension: max - min + 1
        d = coords[0].max() - coords[0].min() + 1
        h = coords[1].max() - coords[1].min() + 1
        w = coords[2].max() - coords[2].min() + 1
        sizes.append((d, h, w))
    
    # Compute average dimensions if valid labels exist
    if sizes:
        sizes = np.array(sizes)
        D_avg = np.mean(sizes[:, 0])
        H_avg = np.mean(sizes[:, 1])
        W_avg = np.mean(sizes[:, 2])
    else:
        print(f"No valid labels found in file {filename}.")
        return

    # Process each error ID in the mask
    for err_id in error_ids:
        coords = np.array(np.where(mask == err_id))
        if coords.size == 0:
            continue  # Skip if the error ID is not present in the mask

        # Choose center based on placement_mode
        if placement_mode == "center":
            center = np.round(coords.mean(axis=1)).astype(int)  # [depth, height, width]
        elif placement_mode == "random":
            random_index = np.random.randint(coords.shape[1])
            center = coords[:, random_index]
        else:
            # Default to center if mode is unknown
            center = np.round(coords.mean(axis=1)).astype(int)
        
        # Remove the original error region by setting it to background (assumed to be 0)
        mask[mask == err_id] = 0
        
        # Ellipsoid parameters: use half of the average dimensions as radii
        rz = D_avg / 2.0
        ry = H_avg / 2.0
        rx = W_avg / 2.0
        
        # Define local bounding box to prevent going out of mask boundaries
        zmin = max(center[0] - int(np.ceil(rz)), 0)
        zmax = min(center[0] + int(np.ceil(rz)) + 1, mask.shape[0])
        ymin = max(center[1] - int(np.ceil(ry)), 0)
        ymax = min(center[1] + int(np.ceil(ry)) + 1, mask.shape[1])
        xmin = max(center[2] - int(np.ceil(rx)), 0)
        xmax = min(center[2] + int(np.ceil(rx)) + 1, mask.shape[2])
        
        # Create a meshgrid for the local region
        zz, yy, xx = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
        # Ellipsoid equation: ((z - cz)/rz)^2 + ((y - cy)/ry)^2 + ((x - cx)/rx)^2 <= 1
        ellipsoid = (((zz - center[0]) / rz) ** 2 +
                     ((yy - center[1]) / ry) ** 2 +
                     ((xx - center[2]) / rx) ** 2) <= 1
        
        # Assign the error ID to voxels inside the ellipsoid in the local region
        mask[zmin:zmax, ymin:ymax, xmin:xmax][ellipsoid] = err_id

    # Save the modified mask to the output folder with the same filename
    output_path = os.path.join(output_folder, filename)
    tifffile.imwrite(output_path, mask)
    print(f"Processed and saved: {output_path}")

def process_folder(input_folder, csv_folder, output_folder, placement_mode):
    """
    Process all 3D mask TIFF files in the input folder.

    For each file, the corresponding CSV file is read (using the csv module) to determine error IDs,
    and the mask is processed accordingly based on the selected placement mode. The modified mask files are saved in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all TIFF files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            filepath = os.path.join(input_folder, filename)
            process_mask_file(filepath, csv_folder, output_folder, placement_mode)

def main():
    # Set up the argument parser for command-line inputs
    parser = argparse.ArgumentParser(
        description="Process 3D mask TIFF files by replacing error label regions with ellipsoids. "
                    "You can choose the placement mode for the ellipsoid: 'center' (default) or 'random'."
    )
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the folder containing 3D mask TIFF files.")
    parser.add_argument("--csv_folder", type=str, required=True,
                        help="Path to the folder containing CSV files with label information.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the folder where processed TIFF files will be saved.")
    parser.add_argument("--placement_mode", type=str, choices=["center", "random"], default="center",
                        help="Mode to choose the ellipsoid placement center: 'center' uses the region center; 'random' selects a random point from the region.")
    
    args = parser.parse_args()
    process_folder(args.input_folder, args.csv_folder, args.output_folder, args.placement_mode)

if __name__ == '__main__':
    main()
