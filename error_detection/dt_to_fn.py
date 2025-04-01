import os
import tifffile
import numpy as np
import csv
from scipy.ndimage import maximum_filter, generate_binary_structure, distance_transform_edt

def get_local_peaks(mask, threshold_ratio=0.5):
    """
    Perform a 3D distance transform on a binary mask and extract local peak coordinates.
    
    Args:
        mask (np.ndarray): 3D binary mask with foreground=1 and background=0.
        threshold_ratio (float): Ratio to filter weak peaks (default is 0.5).
    
    Returns:
        coords (np.ndarray): Array of local peak coordinates (z, y, x).
        dist_transform (np.ndarray): The 3D distance-transformed image.
    """
    # Compute the 3D Euclidean distance transform
    dist_transform = distance_transform_edt(mask)
    
    # Define a 3D neighborhood structure for full connectivity
    neighborhood = generate_binary_structure(3, 2)
    # Find local maxima using the maximum filter
    local_max = (maximum_filter(dist_transform, footprint=neighborhood) == dist_transform)
    
    # Filter out weak peaks based on the threshold ratio
    threshold = threshold_ratio * dist_transform.max()
    peaks = (dist_transform >= threshold) & local_max
    
    # Get coordinates of peak points; output shape: (N, 3) with (z, y, x) ordering
    coords = np.argwhere(peaks)
    return coords, dist_transform

def get_label_list1(labeled_image, peak_coords):
    """
    Retrieve the set of labels from a 3D labeled image at the given peak coordinates.
    
    Args:
        labeled_image (np.ndarray): 3D image where each region is labeled (background=0).
        peak_coords (np.ndarray): Array of peak coordinates (z, y, x).
    
    Returns:
        set: A set of labels (label_list1) corresponding to the peak coordinates.
    """
    labels = set()
    for coord in peak_coords:
        z, y, x = coord
        label_val = labeled_image[z, y, x]
        if label_val > 0:  # Exclude background
            labels.add(label_val)
    return labels

def get_label_list2(csv_file_path, score_threshold=0.5):
    """
    Read a CSV file to extract the second id from the matched_pairs column for rows with 
    a matched score greater than or equal to the threshold.
    
    Args:
        csv_file_path (str): Path to the CSV file.
        score_threshold (float): The score threshold (default is 0.5).
    
    Returns:
        set: A set of unique second ids (label_list2).
    """
    label_list2 = set()
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            score = float(row["matched_scores"])
            if score < score_threshold:
                pair_str = row["matched_pairs"]
                # Remove the surrounding parentheses and split by comma
                pair_str = pair_str.strip("()")
                parts = pair_str.split(',')
                if len(parts) >= 2:
                    # Convert the second part to an integer after stripping spaces
                    second_id = int(parts[1].strip())
                    label_list2.add(second_id)
    return label_list2

def compute_recall(label_list1, label_list2):
    """
    Compute the recall rate of label_list1 with respect to label_list2.
    
    Args:
        label_list1 (set): The set of labels detected from the image.
        label_list2 (set): The set of labels extracted from the CSV file.
    
    Returns:
        tuple: Recall rate (float) and the intersection set of labels.
    """
    intersection = label_list1.intersection(label_list2)
    recall = len(intersection) / len(label_list2) if label_list2 else 0
    return recall, intersection

def process_files(image_folder, mask_folder, csv_folder):
    """
    Process files in the given folders. Each folder is assumed to contain files with the same base name.
    For each file, perform the 3D distance transform-based local peak detection, label extraction,
    CSV processing, and recall computation.
    
    Args:
        image_folder (str): Path to the folder containing 3D labeled images (TIFF files).
        mask_folder (str): Path to the folder containing 3D binary mask images (TIFF files).
        csv_folder (str): Path to the folder containing CSV files.
    """
    # List all CSV files in the csv_folder
    csv_files = [f for f in os.listdir(csv_folder) if f.lower().endswith(".csv")]
    
    for csv_filename in csv_files:
        # Get the base file name (without extension)
        base_name = os.path.splitext(csv_filename)[0]
        # Construct full file paths for image, mask, and csv file (assuming TIFF format for image and mask)
        image_path = os.path.join(image_folder, base_name + ".tif")
        mask_path = os.path.join(mask_folder, base_name + ".tif")
        csv_path = os.path.join(csv_folder, csv_filename)
        
        # Load the 3D mask image using tifffile
        try:
            mask = tifffile.imread(mask_path)
        except Exception as e:
            print(f"Error reading mask file {mask_path}: {e}")
            continue
        
        # Binarize the mask: assume pixel value > 127 as foreground
        mask_bin = (mask > 127).astype(np.uint8)
        
        # Load the 3D labeled image using tifffile
        try:
            labeled_image = tifffile.imread(image_path)
        except Exception as e:
            print(f"Error reading image file {image_path}: {e}")
            continue
        
        # Get local peak coordinates from the binary mask
        peak_coords, _ = get_local_peaks(mask_bin, threshold_ratio=0.5)
        
        # Get label_list1 from the labeled image at the detected peak coordinates
        label_list1 = get_label_list1(labeled_image, peak_coords)
        
        # Extract label_list2 from the CSV file using the csv module
        label_list2 = get_label_list2(csv_path, score_threshold=0.5)
        
        # Compute the recall rate and the matched label intersection
        recall, matched_labels = compute_recall(label_list1, label_list2)
        
        # Print the results for the current file
        print(f"File: {base_name}")
        print("label_list1:", label_list1)
        print("label_list2:", label_list2)
        print("Matched labels (intersection):", matched_labels)
        print(f"Number of matched labels: {len(matched_labels)}")
        print("Recall: {:.2f}".format(recall))
        print("---------------------------")

if __name__ == "__main__":
    # Define the folder paths (adjust as needed)
    image_folder = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/test/images"
    mask_folder = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/test/masks"
    csv_folder = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/proofreaded"
    
    # Process all files in the provided folders
    process_files(image_folder, mask_folder, csv_folder)
