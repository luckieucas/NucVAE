import os
import csv
import numpy as np
import torch
import torch.nn as nn
import tifffile

# ---------------------------
# 3D UNet model definition
# ---------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, init_features=32):
        """
        3D UNet model initialization.
        in_channels: number of input channels (concatenated image and mask)
        out_channels: number of output channels (binary mask output)
        init_features: base number of features for the first layer
        """
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block(features * 8 * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features * 4 * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features * 2 * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _block(in_channels, features, name):
        """
        Creates a convolutional block with two Conv3d layers, each followed by BatchNorm3d and ReLU.
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder pathway
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder pathway with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.sigmoid(self.conv(dec1))

# ---------------------------
# Helper function to compute patch coordinates
# ---------------------------
def get_patch_coords(center, volume_shape, patch_size=(32, 32, 32)):
    """
    Given a center coordinate and the volume shape, compute the start and end indices
    for cropping a patch of size patch_size centered at center. The coordinates are adjusted
    to ensure the patch lies within the volume boundaries.
    
    Parameters:
      center: tuple of (z, y, x)
      volume_shape: shape of the 3D volume (D, H, W)
      patch_size: desired patch size (depth, height, width)
      
    Returns:
      z_start, z_end, y_start, y_end, x_start, x_end
    """
    patch_d, patch_h, patch_w = patch_size
    center_z, center_y, center_x = center

    z_start = max(0, center_z - patch_d // 2)
    y_start = max(0, center_y - patch_h // 2)
    x_start = max(0, center_x - patch_w // 2)

    # Adjust if the patch goes beyond the volume boundaries
    if z_start + patch_d > volume_shape[0]:
        z_start = volume_shape[0] - patch_d
    if y_start + patch_h > volume_shape[1]:
        y_start = volume_shape[1] - patch_h
    if x_start + patch_w > volume_shape[2]:
        x_start = volume_shape[2] - patch_w

    z_end = z_start + patch_d
    y_end = y_start + patch_h
    x_end = x_start + patch_w

    return z_start, z_end, y_start, y_end, x_start, x_end

# ---------------------------
# Function to process a single file
# ---------------------------
def process_file(image_path, mask_path, csv_path, model, device):
    """
    Process a single image/mask pair using the error correction model.
    
    Parameters:
      image_path: path to the image (tif) file
      mask_path: path to the corresponding mask (tif) file
      csv_path: path to the CSV file containing error ids and labels
      model: trained UNet3D model for error correction
      device: torch device (CPU or CUDA)
      
    Returns:
      updated mask (numpy array)
    """
    # Load image and mask volumes using tifffile
    image = tifffile.imread(image_path)  # Expected shape: (D, H, W)
    mask = tifffile.imread(mask_path)    # Expected shape: (D, H, W)

    # Read the CSV file using the built-in csv library to obtain error ids with label==1
    error_ids = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip header row if present
            if row[0].lower() == 'id':
                continue
            try:
                error_id = int(row[0])
                label = int(row[1])
            except:
                continue
            if label == 1:
                error_ids.append(error_id)

    # For each error id, process the corresponding region in the mask
    for error_id in error_ids:
        # Find indices where the mask equals the current error id
        indices = np.where(mask == error_id)
        binary_mask = mask == error_id
        if len(indices[0]) == 0:
            continue  # Skip if no such region exists

        # Remove the error region from the mask
        mask[indices] = 0
        # Compute the center coordinate of the error region
        center_z = int(np.mean(indices[0]))
        center_y = int(np.mean(indices[1]))
        center_x = int(np.mean(indices[2]))
        center = (center_z, center_y, center_x)

        # Compute patch coordinates ensuring a 32x32x32 patch is within the volume
        z_start, z_end, y_start, y_end, x_start, x_end = get_patch_coords(center, mask.shape, patch_size=(32, 32, 32))

        # Extract the patch from both image and mask
        patch_img = image[z_start:z_end, y_start:y_end, x_start:x_end]
        patch_mask = binary_mask[z_start:z_end, y_start:y_end, x_start:x_end]

        # Prepare the input for the model by converting to float32, adding channel dimension,
        # and concatenating along the channel axis to get a 2-channel input.
        patch_img = patch_img.astype(np.float32)
        patch_mask = patch_mask.astype(np.float32)
        #Normalize image and mask
        patch_img = (patch_img - np.min(patch_img)) / (np.max(patch_img) - np.min(patch_img))
        patch_img_tensor = torch.from_numpy(np.expand_dims(patch_img, axis=0))   # Shape: (1, 32, 32, 32)
        patch_mask_tensor = torch.from_numpy(np.expand_dims(patch_mask, axis=0)) # Shape: (1, 32, 32, 32)
        input_tensor = torch.cat([patch_img_tensor, patch_mask_tensor], dim=0)    # Shape: (2, 32, 32, 32)
        input_tensor = input_tensor.unsqueeze(0).to(device)                       # Shape: (1, 2, 32, 32, 32)

        # Run inference using the model
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)  # Output shape: (1, 1, 32, 32, 32)
        output_patch = output.squeeze().cpu().numpy()  # Shape: (32, 32, 32)

        # Threshold the prediction to obtain a binary patch (values 0 or 1)
        predicted_patch = (output_patch > 0.5).astype(np.uint8)

        # Update the corresponding patch region in the mask:
        # For voxels where the prediction is 1, assign the current error id.
        patch_updated = np.where(predicted_patch == 1, error_id, patch_mask)
        # Fill the mask with the current error id for remaining voxels with 0 (background)

    return mask

# ---------------------------
# Testing function that processes all files
# ---------------------------
def test_model(image_folder, mask_folder, output_folder, error_csv_folder):
    """
    Processes each file in the image folder:
      - Loads the image and its corresponding mask and CSV file.
      - Extracts error regions and performs patch-based correction using the model.
      - Saves the updated mask to the output folder with the same filename.
    
    Parameters:
      image_folder: directory containing image (tif) files
      mask_folder: directory containing mask (tif) files (filenames should match image files)
      output_folder: directory where corrected mask files will be saved
      error_csv_folder: directory containing CSV files with error info (same base filename as image)
    """
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model and load the trained weights
    model = UNet3D(in_channels=2, out_channels=1, init_features=32)
    model_path = "/mmfs1/data/liupen/project/NucleiVae/expriments/unet3d_correction/best_unet3d_model.pth"  # Adjust path if necessary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each TIFF image file in the image folder
    for filename in os.listdir(image_folder):
        if not filename.endswith('.tif'):
            continue
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        # Assume CSV file has the same base filename but with .csv extension
        csv_path = os.path.join(error_csv_folder, filename.replace('.tif', '.csv'))
        
        if not os.path.exists(mask_path) or not os.path.exists(csv_path):
            print(f"Missing corresponding mask or CSV for {filename}. Skipping.")
            continue

        print(f"Processing file: {filename}")
        updated_mask = process_file(image_path, mask_path, csv_path, model, device)
        output_path = os.path.join(output_folder, filename)
        tifffile.imwrite(output_path, updated_mask)
        print(f"Saved updated mask to {output_path}")

if __name__ == '__main__':
    # Example usage: specify your folders here
    image_folder = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/test/images"      # Folder containing TIFF image files
    mask_folder = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/test/masks"          # Folder containing TIFF mask files
    output_folder = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/test/masks/unet_corrected"      # Folder to save the corrected masks
    error_csv_folder = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/test/masks/extracted_match_score" # Folder containing CSV files with error info
    
    test_model(image_folder, mask_folder, output_folder, error_csv_folder)
