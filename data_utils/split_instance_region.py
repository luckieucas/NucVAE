import numpy as np
import tifffile
import argparse
import os

def split_mask(input_file):
    """
    Splits a 3D binary mask into four equal regions based on its bounding box.
    Each new mask contains only one of the four regions, while the rest is set to 0.

    Args:
        input_file (str): Path to the input 3D TIFF mask file.
    """
    # Load the TIFF file (expected shape: (z, height, width))
    mask = tifffile.imread(input_file)

    # Get the foreground (non-zero) pixel indices to determine the bounding box
    z_inds, y_inds, x_inds = np.where(mask > 0)
    if z_inds.size == 0:
        raise ValueError("The input mask contains no foreground (non-zero) pixels!")

    # Compute the bounding box limits
    zmin, zmax = z_inds.min(), z_inds.max()
    ymin, ymax = y_inds.min(), y_inds.max()
    xmin, xmax = x_inds.min(), x_inds.max()

    # Compute the center points for y and x to divide the bounding box into four parts
    y_center = (ymin + ymax) // 2
    x_center = (xmin + xmax) // 2

    # Generate output file names by appending region names to the input file name
    base_name, ext = os.path.splitext(input_file)
    output_files = {
        "top_left": f"{base_name}_top_left{ext}",
        "top_right": f"{base_name}_top_right{ext}",
        "bottom_left": f"{base_name}_bottom_left{ext}",
        "bottom_right": f"{base_name}_bottom_right{ext}",
    }

    # Create four new masks, each preserving only one region of the bounding box

    # Region 1: Top-left (y: ymin ~ y_center, x: xmin ~ x_center)
    mask_top_left = np.zeros_like(mask)
    mask_top_left[zmin:zmax+1, ymin:y_center+1, xmin:x_center+1] = mask[zmin:zmax+1, ymin:y_center+1, xmin:x_center+1]
    tifffile.imwrite(output_files["top_left"], mask_top_left)

    # Region 2: Top-right (y: ymin ~ y_center, x: x_center+1 ~ xmax)
    mask_top_right = np.zeros_like(mask)
    mask_top_right[zmin:zmax+1, ymin:y_center+1, x_center+1:xmax+1] = mask[zmin:zmax+1, ymin:y_center+1, x_center+1:xmax+1]
    tifffile.imwrite(output_files["top_right"], mask_top_right)

    # Region 3: Bottom-left (y: y_center+1 ~ ymax, x: xmin ~ x_center)
    mask_bottom_left = np.zeros_like(mask)
    mask_bottom_left[zmin:zmax+1, y_center+1:ymax+1, xmin:x_center+1] = mask[zmin:zmax+1, y_center+1:ymax+1, xmin:x_center+1]
    tifffile.imwrite(output_files["bottom_left"], mask_bottom_left)

    # Region 4: Bottom-right (y: y_center+1 ~ ymax, x: x_center+1 ~ xmax)
    mask_bottom_right = np.zeros_like(mask)
    mask_bottom_right[zmin:zmax+1, y_center+1:ymax+1, x_center+1:xmax+1] = mask[zmin:zmax+1, y_center+1:ymax+1, x_center+1:xmax+1]
    tifffile.imwrite(output_files["bottom_right"], mask_bottom_right)

    print("Four new mask files have been successfully saved:")
    for region, filename in output_files.items():
        print(f"  - {region}: {filename}")

def main():
    """
    Parses command-line arguments and processes the input 3D binary mask file.
    """
    parser = argparse.ArgumentParser(
        description="Splits a 3D binary mask into four regions based on its bounding box."
    )
    parser.add_argument("--input_file", help="Path to the input TIFF file")
    
    args = parser.parse_args()
    
    try:
        split_mask(args.input_file)
    except Exception as e:
        print(f"Error occurred during processing: {e}")

if __name__ == "__main__":
    main()
