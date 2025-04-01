import os
import argparse
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from scipy.ndimage import label
from em_util.seg import seg_to_iou
from tqdm import tqdm


from vae_eval import compute_log_likelihood
from data_utils import extract_patch_centered
from models import DualEncoderVAE3D
from test import vae_inference
from evaluate import evaluate_res

# new label list
LABEL_LIST = [i for i in range(1001, 2000)]

# =============================================================================
# Note: It is assumed that the following classes and functions are defined/imported:
#   - VAE3D and DualEncoderVAE3D model classes
#   - compute_log_likelihood: to compute the log-likelihood of an instance using a VAE
#   - vae_inference: to perform inference and reconstruction using a VAE model
# =============================================================================

def largest_cc(seg):
    """
    Extract the largest connected component from a 3D segmentation.
    
    Args:
        seg (numpy.ndarray): 3D array of shape (D, H, W) with foreground as nonzero.
    
    Returns:
        numpy.ndarray: The largest connected component of the 3D segmentation.
    """
    labeled, num_features = label(seg)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background
    largest_label = counts.argmax()

    largest_cc = (labeled == largest_label)
    return largest_cc.astype(seg.dtype) 


def extract_patch(mask, bbox, target_shape):
    """
    Extract the center patch of the 3D mask.
    
    Args:
        mask (numpy.ndarray): 3D array of shape (D, H, W) with foreground as nonzero.
        target_shape (tuple): Desired output shape (D, H, W).
    
    Returns:
        numpy.ndarray: The center path of the 3D mask with shape target_shape.
    """
    center = []
    for d, s in enumerate(target_shape):
        vol_size = mask.shape[d]
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
        return None
    center = tuple(center)
    return extract_patch_centered(mask, center, target_shape)


def compute_bounding_box(mask):
    """
    Compute the bounding box of the foreground in a 3D mask.
    
    Args:
        mask (numpy.ndarray): 3D array of shape (D, H, W) with foreground as nonzero.
    
    Returns:
        tuple: (z_min, z_max, y_min, y_max, x_min, x_max) bounding box,
               or None if no foreground is found.
    """
    indices = np.where(mask)
    if len(indices[0]) == 0:
        return None
    z_min, y_min, x_min = np.min(indices[0]), np.min(indices[1]), np.min(indices[2])
    z_max, y_max, x_max = np.max(indices[0]), np.max(indices[1]), np.max(indices[2])
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def split_bbox_into_4(bbox):
    """
    Split a 3D bounding box into 4 subregions by dividing the x and y axes into two parts.
    The z-axis remains unchanged.
    
    Args:
        bbox (tuple): A tuple representing the bounding box in the format 
                      (z_min, z_max, y_min, y_max, x_min, x_max)
                      
    Returns:
        list of tuples: A list containing 4 sub-boxes, each in the same format as the input bbox.
    """
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    # Compute midpoints for the y and x axes.
    y_mid = (y_min + y_max) // 2
    x_mid = (x_min + x_max) // 2

    # Define two partitions for y and x using inclusive ranges.
    y_parts = [(y_min, y_mid), (y_mid + 1, y_max)]
    x_parts = [(x_min, x_mid), (x_mid + 1, x_max)]

    sub_boxes = []
    # The z-axis remains unchanged.
    for yp in y_parts:
        for xp in x_parts:
            sub_boxes.append((z_min, z_max, yp[0], yp[1], xp[0], xp[1]))
    return sub_boxes

def split_bbox_into_8(bbox):
    """
    Split a 3D bounding box into 8 subregions by dividing each axis into two parts.
    
    Args:
        bbox (tuple): A tuple representing the bounding box in the format 
                      (z_min, z_max, y_min, y_max, x_min, x_max)
                      
    Returns:
        list of tuples: A list containing 8 sub-boxes, each in the same format as the input bbox.
    """
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    # Compute midpoints for each axis
    z_mid = (z_min + z_max) // 2
    y_mid = (y_min + y_max) // 2
    x_mid = (x_min + x_max) // 2

    # Define the two partitions for each axis (using inclusive ranges)
    z_parts = [(z_min, z_mid), (z_mid + 1, z_max)]
    y_parts = [(y_min, y_mid), (y_mid + 1, y_max)]
    x_parts = [(x_min, x_mid), (x_mid + 1, x_max)]

    sub_boxes = []
    for zp in z_parts:
        for yp in y_parts:
            for xp in x_parts:
                sub_boxes.append((zp[0], zp[1], yp[0], yp[1], xp[0], xp[1]))
    return sub_boxes

def region_center(bbox):
    """
    Calculate the center coordinates of a given bounding box.
    
    Args:
        bbox (tuple): (z_min, z_max, y_min, y_max, x_min, x_max)
    
    Returns:
        tuple: Center coordinates (z, y, x)
    """
    z_center = (bbox[0] + bbox[1]) // 2
    y_center = (bbox[2] + bbox[3]) // 2
    x_center = (bbox[4] + bbox[5]) // 2
    return (z_center, y_center, x_center)

def crop_patch_with_padding(volume, center, patch_size=(32, 32, 32)):
    """
    Crop a patch of given size centered at 'center' from a 3D volume.
    Pads with zeros if the patch goes beyond the volume boundaries.
    
    Args:
        volume (numpy.ndarray): 3D array of shape (D, H, W).
        center (tuple): Center coordinates (z, y, x).
        patch_size (tuple): Desired patch size (D, H, W).
    
    Returns:
        numpy.ndarray: Cropped patch of shape patch_size.
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    cz, cy, cx = center
    # Compute start and end indices
    z_start = cz - pd // 2
    y_start = cy - ph // 2
    x_start = cx - pw // 2
    z_end = z_start + pd
    y_end = y_start + ph
    x_end = x_start + pw

    # Calculate necessary padding amounts
    pad_before = [max(0, -z_start), max(0, -y_start), max(0, -x_start)]
    pad_after  = [max(0, z_end - D), max(0, y_end - H), max(0, x_end - W)]
    padded = np.pad(volume, ((pad_before[0], pad_after[0]),
                             (pad_before[1], pad_after[1]),
                             (pad_before[2], pad_after[2])), mode='constant')
    # Adjust indices after padding
    z_start += pad_before[0]
    y_start += pad_before[1]
    x_start += pad_before[2]
    patch = padded[z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
    return patch

def compute_iou(seg1, seg2):
    """
    Compute the Intersection over Union (IoU) for two binary segmentation volumes.
    
    Args:
        seg1 (numpy.ndarray): First segmentation volume.
        seg2 (numpy.ndarray): Second segmentation volume.
    
    Returns:
        float: IoU value.
    """
    intersection = np.logical_and(seg1, seg2).sum()
    union = np.logical_or(seg1, seg2).sum()
    return intersection / union if union > 0 else 0


def load_dual_vae_model(model_path, device):
    """
    Load a pre-trained dual encoder VAE model.
    
    Args:
        model_path (str): Path to the dual encoder VAE model weights.
        device (torch.device): Device to load the model on.
    
    Returns:
        model: Loaded dual encoder VAE model.
    """
    model = DualEncoderVAE3D(mask_in_channels=1, img_in_channels=2, latent_dim=16, base_channel=16).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_instance(instance_mask, image, dual_vae_model, ll_threshold,  overlap_thresh, global_mask, save_path, gt_mask,filter_size=200):
    """
    Process a single instance by:
      1. Computing its log-likelihood (using a patch extracted from the full instance).
      2. If the likelihood is below the threshold, splitting the instance's bounding box 
         into 8 subregions (dividing each axis into 2 parts).
      3. For each subregion, extract a 32x32x32 patch (from both image and mask) centered 
         on the foreground, and reconstruct the patch using the dual encoder VAE.
      4. Threshold the reconstruction to obtain a binary segmentation patch.
      5. For each predicted patch, extract its corresponding region from a modified global mask 
         (with the current instance removed) and compute IoU via seg_to_iou.
         If the IoU exceeds overlap_thresh, merge the predicted patch into the global mask.
    
    Args:
        instance_mask (numpy.ndarray): 3D binary mask for the instance (D, H, W).
        image (numpy.ndarray): Corresponding 3D image (D, H, W).
        dual_vae_model: Dual encoder VAE model for reconstruction.
        ll_threshold (float): Log-likelihood threshold.
        filter_size (int): Minimum size threshold for filtering segments (used by seg_to_iou).
        overlap_thresh (float): IoU threshold for merging predicted patches.
        global_mask (numpy.ndarray): Global mask containing all instances.
        save_path (str): Path to save the reconstructed patches.
        
    Returns:
        numpy.ndarray: The modified global mask after merging the accepted predictions.
    """
    # Compute the bounding box of the instance.
    bbox = compute_bounding_box(instance_mask)
    if bbox is None:
        return global_mask

    # Extract a 32x32x32 patch from the instance mask (using the full bounding box)
    instance_patch = extract_patch(instance_mask, bbox, (32, 32, 32))
    if instance_patch is None:
        return global_mask
    print(f"instance_patch shape: {instance_patch.shape}")

    # Compute log-likelihood using the extracted patch.
    instance_patch_tensor = torch.from_numpy(instance_patch).type(torch.float32).unsqueeze(0).cuda()
    ll = compute_log_likelihood(dual_vae_model, instance_patch_tensor)
    if ll >= ll_threshold:
        print(f"Instance log-likelihood {ll:.4f} >= threshold {ll_threshold}. Skipping further processing.")
        return global_mask
    print(f"Instance log-likelihood {ll:.4f} < threshold {ll_threshold}. Proceeding with processing.")

    # Split the instance's bounding box into 8 subregions (dividing each axis into 2 parts)
    region_bboxes = split_bbox_into_4(bbox)
    
    # Create a modified copy of the global mask by setting the current instance region to background.
    modified_global_mask = global_mask.copy()
    modified_global_mask[instance_mask == 1] = 0

    # Process each of the 8 subregions.
    mask_merged = False
    print(f"Total volume of current instance: {np.sum(instance_mask)}")
    for reg_bbox in region_bboxes:
        # Extract subregion from instance_mask using the sub-box coordinates.
        z0, z1, y0, y1, x0, x1 = reg_bbox
        region_mask = instance_mask[z0:z1+1, y0:y1+1, x0:x1+1]
        masked_instance_mask = np.zeros_like(instance_mask)
        masked_instance_mask[z0:z1+1, y0:y1+1, x0:x1+1] = region_mask

        # Compute the foreground bounding box within the current subregion.
        reg_bbox_fg = compute_bounding_box(region_mask)
        if reg_bbox_fg is None:
            continue

        # Convert the subregion's foreground bounding box to global coordinates.
        reg_bbox_fg_full = (reg_bbox_fg[0] + z0, reg_bbox_fg[1] + z0,
                            reg_bbox_fg[2] + y0, reg_bbox_fg[3] + y0,
                            reg_bbox_fg[4] + x0, reg_bbox_fg[5] + x0)
        # Calculate the center (global coordinates) of this foreground box.
        center = region_center(reg_bbox_fg_full)

        # Crop a 32x32x32 patch from the image and instance mask centered at the computed center.
        patch_image = crop_patch_with_padding(image, center, patch_size=(32, 32, 32))
        patch_mask  = crop_patch_with_padding(masked_instance_mask, center, patch_size=(32, 32, 32))
        # Extract largest connected component from the patch mask.
        patch_mask = largest_cc(patch_mask)

        # Skip processing if the extracted patch is too small.
        if patch_mask.sum() < filter_size:
            print(f"Patch size {patch_mask.sum()} < {filter_size}. Skipping further processing.")
            continue
        # Prepare dual encoder input: channel 0 is normalized image patch, channel 1 is mask patch.
        patch_image_tensor = torch.from_numpy(patch_image).float().unsqueeze(0)  # shape: (1, 32, 32, 32)
        patch_mask_tensor  = torch.from_numpy(patch_mask).float().unsqueeze(0)   # shape: (1, 32, 32, 32)
        patch_image_tensor = patch_image_tensor / 255.0  # Normalize assuming [0, 255] range.
        dual_input = torch.cat([patch_image_tensor, patch_mask_tensor], dim=0)  # shape: (2, 32, 32, 32)

        # Perform reconstruction using the dual encoder VAE (using the image branch: use_mask_encoder=False)
        device = next(dual_vae_model.parameters()).device
        dual_input = dual_input.to(device)
        _, recon = vae_inference(dual_vae_model, dual_input, device, use_mask_encoder=False)
        recon_np = recon.cpu().numpy()[0, 0]  # Extract the first channel.
        
        # compute log-likelihood using the extracted patch.
        ll = compute_log_likelihood(dual_vae_model, (recon[0]>0.5).type(torch.float32))
        print(f"Reconstruction log-likelihood: {ll:.4f}")
        if ll < -200:
            continue
        
        # Save patch image, mask, and reconstruction.
        patch_image_path = save_path.replace(".tif","") + f"patch_image_{center[0]}_{center[1]}_{center[2]}.tif"
        patch_mask_path = save_path.replace(".tif","") + f"patch_mask_{center[0]}_{center[1]}_{center[2]}.tif"
        recon_path = save_path.replace(".tif","") + f"recon_{center[0]}_{center[1]}_{center[2]}.tif"
        tifffile.imwrite(patch_image_path, patch_image)
        tifffile.imwrite(patch_mask_path, patch_mask)
        tifffile.imwrite(recon_path, recon_np)
        
        # Threshold the reconstruction to obtain a binary segmentation patch (values > 0.5 become 1).
        seg_patch = (recon_np > 0.5).astype(np.uint8)

        # Compute global placement indices for the patch.
        pd, ph, pw = seg_patch.shape  # Expected shape: (32, 32, 32)
        z_start = center[0] - pd // 2
        y_start = center[1] - ph // 2
        x_start = center[2] - pw // 2
        z_end = z_start + pd
        y_end = y_start + ph
        x_end = x_start + pw

        # Clip the indices to ensure they are within the bounds of the instance mask.
        D, H, W = instance_mask.shape
        z_start_clip = max(z_start, 0)
        y_start_clip = max(y_start, 0)
        x_start_clip = max(x_start, 0)
        z_end_clip = min(z_end, D)
        y_end_clip = min(y_end, H)
        x_end_clip = min(x_end, W)

        # Extract the corresponding region from the modified global mask.
        global_patch = modified_global_mask[z_start_clip:z_end_clip,
                                            y_start_clip:y_end_clip,
                                            x_start_clip:x_end_clip]
        # For IOU computation, use the predicted patch (cropped to the same region)
        pred_patch = seg_patch[
            z_start_clip - z_start: z_end_clip - z_start,
            y_start_clip - y_start: y_end_clip - y_start,
            x_start_clip - x_start: x_end_clip - x_start
        ]
        # Here, seg_to_iou is assumed to return a list of tuples:
        # (predicted_label, mask_label, predicted_area, mask_area, intersection_area)
        # For binary patches, the label is simply 1 when foreground is present.
        # We use pred_patch and global_patch to compute IoU.
        merged_region = np.zeros_like(global_patch, dtype=np.uint16)
        if np.sum(global_patch) == 0:
            merge_label = LABEL_LIST.pop(0)
            merged_region = pred_patch*merge_label
        else:
            iou_pairs = seg_to_iou(pred_patch, global_patch)
            merge_patch = False
            for pair in iou_pairs:
                if pair[1] != 0:  # There is a corresponding region in the global mask.
                    denom = pair[2] + pair[3] - pair[4]
                    iou = pair[4] / denom if denom > 0 else 0
                    print(f"Patch IoU: {iou:.2f}")
                    if iou > overlap_thresh:
                        merge_patch = True
                        print(f"Merging patch at center {center} with IoU {iou:.2f}")
                        break

            # If the IoU exceeds the threshold, merge the predicted patch into the global mask.
            if not merge_patch:
                merge_label = LABEL_LIST.pop(0)
                pred_patch = pred_patch*merge_label
                # Update the global mask region with the union (logical OR) of the current global region and the prediction.
                global_region = modified_global_mask[z_start_clip:z_end_clip,
                                            y_start_clip:y_end_clip,
                                            x_start_clip:x_end_clip]
                merged_region = np.maximum(global_region, pred_patch).astype(np.uint16)

        # Update the global mask with the merged region.
        merged_region = merged_region.astype(np.uint16)
        modified_global_mask[z_start_clip:z_end_clip,
                    y_start_clip:y_end_clip,
                    x_start_clip:x_end_clip] = merged_region
        mask_merged = True
    if mask_merged:
        F1 = 0
        if gt_mask is not None:
            metrics = evaluate_res(gt_mask, modified_global_mask)
            print(f"evaluate res: {metrics}")
            F1 = metrics['f1']
        
        tifffile.imwrite(save_path.replace(".tif","") + f"_F1_{F1:.2f}.tif", modified_global_mask.astype(np.uint16))

    # else:
    #     modified_global_mask = global_mask
    # Return the modified global mask.
    return modified_global_mask


def main_pipeline(image_path, mask_path, gt_path, output_dir, dual_vae_model_path, ll_threshold):
    """
    Main processing pipeline that loads a 3D image and its corresponding mask, 
    extracts individual instances, processes each instance, and saves the final segmentation patches.
    
    Args:
        image_path (str): Path to the 3D image TIFF file.
        mask_path (str): Path to the 3D mask TIFF file.
        gt_path (str): Path to the ground truth mask TIFF file.
        vae_model_path (str): Path to the VAE model weights for log-likelihood estimation.
        dual_vae_model_path (str): Path to the dual encoder VAE model weights for reconstruction.
        ll_threshold (float): Log-likelihood threshold.
    """
    # Load the 3D image and mask (TIFF format)
    image = tifffile.imread(image_path)  # shape (D, H, W)
    mask = tifffile.imread(mask_path)      # shape (D, H, W)
    if gt_path is not None:
        gt_mask = tifffile.imread(gt_path)
    else:
        gt_mask = None
    base_name, ext = os.path.splitext(mask_path)
    base_name = os.path.basename(base_name)
    dir_name = os.path.dirname(mask_path)

    # Use connected components to label individual instances (assumes binary mask)
    unique_values = np.unique(mask)
    num_features = len(unique_values) - 1
    print(f"Detected {num_features} instances in the mask.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_vae_model = load_dual_vae_model(dual_vae_model_path, device)

    total_final_patches = []
    final_mask = mask.copy()
    for label in unique_values[1:]:
        print(f"\nProcessing instance {label} ...")
        instance_mask = (mask == label).astype(np.uint8)
        # filter out small instances
        if np.sum(instance_mask) < 200:
            print(f"Instance {label} is too small. Skipping.")
            final_mask[instance_mask==1] = 0
            continue
        save_path = f"{output_dir}/{base_name}_instance_{label}_rec.tif"
        final_mask = process_instance(
            instance_mask, image, dual_vae_model, 
            ll_threshold, 0.2,final_mask,save_path,gt_mask
            )
        # Save each segmentation patch after processing
    print(f"\nTotal {len(total_final_patches)} final segmentation patches generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Image & Mask Instance Processing using VAE")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input 3D image TIFF file")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the input 3D mask TIFF file")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the input 3D gt TIFF file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--dual_vae_model_path", type=str, required=True, help="Path to the dual encoder VAE model weights for reconstruction")
    parser.add_argument("--ll_threshold", type=float, default=-1000, help="Log-likelihood threshold")
    args = parser.parse_args()

    # make the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Call the main processing pipeline for each file
    for image_file in tqdm(os.listdir(args.image_path)):
        if image_file.endswith(".tif"):
            image_path = os.path.join(args.image_path, image_file)
            mask_path = os.path.join(args.mask_path, image_file)
            gt_path = os.path.join(args.gt_path, image_file)
            main_pipeline(image_path, mask_path, gt_path, args.output_dir, args.dual_vae_model_path, args.ll_threshold)
