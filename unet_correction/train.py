import os
import random
import numpy as np

import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------------
# Data augmentation functions
# ---------------------------
def noise_transform(image, noise_std=0.01):
    """
    Adds Gaussian noise to the image.
    """
    noise = np.random.normal(0, noise_std, image.shape)
    return image + noise

def random_mask_drop(mask, drop_prob=0.5, drop_ratio=0.1):
    """
    Randomly drops a small block in the mask.
    
    Parameters:
      drop_prob: probability to drop a block
      drop_ratio: size of dropped block relative to overall mask dimensions
    """
    if random.random() < drop_prob:
        d, h, w = mask.shape
        drop_d = max(1, int(d * drop_ratio))
        drop_h = max(1, int(h * drop_ratio))
        drop_w = max(1, int(w * drop_ratio))
        d_start = random.randint(0, d - drop_d) if d > drop_d else 0
        h_start = random.randint(0, h - drop_h) if h > drop_h else 0
        w_start = random.randint(0, w - drop_w) if w > drop_w else 0
        mask[d_start:d_start+drop_d, h_start:h_start+drop_h, w_start:w_start+drop_w] = 0
    return mask

def random_mask_edge_increase(mask, add_prob=0.5, edge_width=3):
    """
    Increases the mask values (sets to 1) along the edges.
    
    Parameters:
      add_prob: probability to perform edge increase
      edge_width: width of the edge to increase
    """
    if random.random() < add_prob:
        d, h, w = mask.shape
        # Top and bottom faces
        mask[:edge_width, :, :] = 1
        mask[-edge_width:, :, :] = 1
        # Left and right faces
        mask[:, :edge_width, :] = 1
        mask[:, -edge_width:, :] = 1
        # Front and back faces
        mask[:, :, :edge_width] = 1
        mask[:, :, -edge_width:] = 1
    return mask

# ---------------------------
# Dataset class for error correction
# ---------------------------
class ErrorCorrectionDataset(Dataset):
    def __init__(self, data_dir, transform=True, val=False):
        """
        data_dir: directory containing data files, assumed to have paired image and mask files.
        transform: whether to apply data augmentation (True for training)
        val: flag indicating if the dataset is used for validation (different augmentation)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.val = val
        # Get all image files (assumes filenames contain "image")
        self.image_files = sorted([f for f in os.listdir(data_dir) if "image" in f and f.endswith("tif")])
        print(f"Total training data:{len(self.image_files)}")
        if len(self.image_files) == 0:
            raise ValueError("No files containing 'image' found in the directory.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and corresponding mask. Assumes mask file is named by replacing "image" with "mask"
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_filename)
        mask_path = image_path.replace("image", "mask")
        image = tiff.imread(image_path)  # Expected shape: (D, H, W)
        mask = tiff.imread(mask_path)    # Expected shape: (D, H, W) as binary
        
        # Keep a copy of the original mask as target
        target_mask = mask.copy()
        # Normalize image and mask
        image = (image - image.min()) / (image.max() - image.min())
        # Apply data augmentations
        if self.transform and not self.val:
            image = noise_transform(image, noise_std=0.01)
            mask = random_mask_drop(mask, drop_prob=0.7, drop_ratio=0.8)
            mask = random_mask_edge_increase(mask, add_prob=0.5, edge_width=5)
        elif self.val:
            # For validation, only lightly drop a small block from the input mask; target remains unchanged
            mask = random_mask_drop(mask.copy(), drop_prob=0.5, drop_ratio=0.3)
        
        # Convert to float32 and add channel dimension
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        target_mask = target_mask.astype(np.float32)
        image = np.expand_dims(image, axis=0)  # (1, D, H, W)
        mask = np.expand_dims(mask, axis=0)    # (1, D, H, W)
        target_mask = np.expand_dims(target_mask, axis=0)
        # Concatenate image and mask to form input with 2 channels
        input_tensor = np.concatenate([image, mask], axis=0)
        
        return torch.from_numpy(input_tensor), torch.from_numpy(target_mask)

# ---------------------------
# 3D UNet model definition
# ---------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, init_features=32):
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
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

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
# Training process with model saving
# ---------------------------
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, save_path):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iteration = 0
        for inputs, targets in train_loader:
            if iteration >= 50:
                break
            iteration += 1
            inputs = inputs.to(device)     # (batch, 2, D, H, W)
            targets = targets.to(device)   # (batch, 1, D, H, W)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")

        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    # Directory containing training data
    data_dir = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/train/unet_correction_data/images"  # Replace with your data folder path

    # Initialize dataset; transform=True for training data augmentation,
    # and val flag for validation data augmentation
    full_dataset = ErrorCorrectionDataset(data_dir, transform=True, val=False)
    dataset_size = len(full_dataset)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Set val flag for the validation dataset
    val_dataset.dataset.val = True

    # DataLoader settings
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # Model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=2, out_channels=1, init_features=32)
    # Using Binary Cross-Entropy Loss; you may also consider using Dice Loss or others
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 1000
    save_path = "/mmfs1/data/liupen/project/NucleiVae/expriments/unet3d_correction/best_unet3d_model.pth"  # Path to save the best model

    # Start training
    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, save_path)
    # Save final model
    torch.save(model.state_dict(), "/mmfs1/data/liupen/project/NucleiVae/expriments/unet3d_correction/final_unet3d_model.pth")