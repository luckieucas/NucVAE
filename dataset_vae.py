import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio 
import tifffile as tiff


# 新增自定义 transform：随机在前景区域 mask 掉一个小块
class RandomForegroundMask(tio.transforms.Transform):
    def __init__(self, block_size=(4,4,4), p=0.5):
        """
        Args:
            block_size (tuple): 掩盖区域的尺寸 (D, H, W)
            p (float): 该变换的概率
        """
        super().__init__()
        self.block_size = block_size
        self.p = p

    def __call__(self, subject):
        if random.random() >= self.p:
            return subject  # 不做改变
        return self.apply_transform(subject)

    def apply_transform(self, subject):
        # 针对 'mask' 进行变换，假设其为 LabelMap，数据 shape (C, D, H, W)
        mask_tensor = subject['mask'].data  # torch.Tensor
        C, D, H, W = mask_tensor.shape
        # 找到前景位置（非0值）
        foreground_coords = (mask_tensor > 0).nonzero(as_tuple=False)
        if foreground_coords.size(0) == 0:
            return subject
        # 随机选取一个前景 voxel 作为中心
        idx = random.randint(0, foreground_coords.size(0)-1)
        _, z, y, x = foreground_coords[idx]
        z, y, x = z.item(), y.item(), x.item()
        half_D = self.block_size[0] // 2
        half_H = self.block_size[1] // 2
        half_W = self.block_size[2] // 2
        z_start = max(z - half_D, 0)
        y_start = max(y - half_H, 0)
        x_start = max(x - half_W, 0)
        z_end = min(z_start + self.block_size[0], D)
        y_end = min(y_start + self.block_size[1], H)
        x_end = min(x_start + self.block_size[2], W)
        # 将该区域置为 0
        subject['mask'].data[:, z_start:z_end, y_start:y_end, x_start:x_end] = 0
        return subject

def load_tiff_as_numpy(filepath):
    """Load a 3D TIFF file as a NumPy array."""
    return tiff.imread(filepath).astype(np.uint16)

def extract_bounding_box_3d(mask_3d, label_id, min_foreground_pixels=200):
    """
    Extract the bounding box region for a given instance (label_id) from a 3D mask.

    Args:
        mask_3d (np.ndarray): 3D segmentation mask.
        label_id (int): The target instance ID.
        min_foreground_pixels (int): Minimum number of foreground pixels required.

    Returns:
        tuple or None: (cropped_mask, bbox) if valid, where cropped_mask is the binary mask crop,
                       and bbox is a tuple (zmin, zmax, ymin, ymax, xmin, xmax).
                       Returns None if the instance is too small.
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

    # Convert to binary mask: 1 for the instance, 0 for background
    cropped[cropped != label_id] = 0
    cropped[cropped == label_id] = 1

    if np.sum(cropped) < min_foreground_pixels:
        return None

    bbox = (zmin, zmax, ymin, ymax, xmin, xmax)
    return cropped, bbox

def extract_patch_centered(volume, center, target_shape, pad_mode='constant', constant_values=0):
    """
    Extract a patch of target_shape from the volume centered at the given center.
    If the patch goes out of the volume bounds, pad the patch to achieve the target shape.

    Args:
        volume (np.ndarray): 3D volume to extract the patch from.
        center (tuple): Center coordinates (z, y, x).
        target_shape (tuple): Desired patch shape (D, H, W).
        pad_mode (str): Padding mode (default is 'constant').
        constant_values (int): Constant value for padding if pad_mode is 'constant'.

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

class InstanceMaskBBoxDataset(Dataset):
    """
    Dataset that loads 3D mask TIFF files and corresponding image files.
    For each randomly chosen instance, it extracts the instance's bounding box from the mask,
    then uses a center location to crop a volume of target_shape from both the mask and the image.
    In training mode (train=True) the crop center is randomly选择，在测试模式 (train=False) 则选取合法区域内的中心点 (center patch).
    """
    def __init__(self,
                 mask_dir,
                 image_dir=None,
                 target_shape=(32, 32, 32),
                 ignore_label_zero=True,
                 min_foreground_pixels=300,
                 max_retries=10,
                 augment=None,
                 train=True):
        """
        Args:
            mask_dir (str): Directory containing 3D TIFF mask files.
            image_dir (str or None): Directory containing 3D TIFF image files.
                If provided, it is assumed that mask and image files share the same filename.
            target_shape (tuple): Desired output shape (D, H, W) for both mask and image.
            ignore_label_zero (bool): If True, ignore the background label (0).
            min_foreground_pixels (int): Minimum number of foreground pixels required.
            max_retries (int): Maximum number of attempts to find a valid instance.
            augment: Optional augmentation transform (torchio).
            train (bool): 如果为 True，则随机裁剪 patch；如果为 False，则采用中心裁剪 (center patch)。
        """
        super().__init__()
        self.mask_dir = Path(mask_dir)
        self.target_shape = target_shape
        self.ignore_label_zero = ignore_label_zero
        self.min_foreground_pixels = min_foreground_pixels
        self.max_retries = max_retries
        self.augment = augment
        self.train = train

        self.mask_paths = sorted(self.mask_dir.glob("*.tif"))
        if len(self.mask_paths) == 0:
            raise ValueError(f"No TIFF files found in {self.mask_dir}")

        if image_dir is not None:
            self.image_dir = Path(image_dir)
            self.image_paths = {p.stem: p for p in sorted(self.image_dir.glob("*.tif"))}
        else:
            self.image_dir = None
            self.image_paths = {}

    def __len__(self):
        return len(self.mask_paths) * 5

    def __getitem__(self, _):
        """
        Randomly selects an instance and returns:
            mask_tensor: Cropped and padded mask tensor (1, D, H, W) (binary: 1 for instance, 0 for background)
            image_tensor: Cropped image patch from the original volume (1, D, H, W)
        Both patches are extracted using a center location that is either randomly chosen (training mode)
        or fixed as the center of the valid range (test mode).
        """
        for _ in range(self.max_retries):
            mask_path = random.choice(self.mask_paths)
            mask_volume = load_tiff_as_numpy(mask_path)
            file_stem = mask_path.stem

            if self.image_dir is not None:
                if file_stem in self.image_paths:
                    image_volume = load_tiff_as_numpy(self.image_paths[file_stem])
                else:
                    continue
            else:
                image_volume = None

            unique_ids = np.unique(mask_volume)
            if self.ignore_label_zero:
                unique_ids = unique_ids[unique_ids != 0]
            if len(unique_ids) == 0:
                continue

            label_id = random.choice(unique_ids)
            result = extract_bounding_box_3d(mask_volume, label_id, self.min_foreground_pixels)
            if result is None:
                continue
            _, bbox = result
            # bbox: (zmin, zmax, ymin, ymax, xmin, xmax)
            
            valid_center = []
            # For each spatial dimension (z, y, x)
            for d, s in enumerate(self.target_shape):
                vol_size = mask_volume.shape[d]
                half = s // 2
                # Get bounding box min and max for current dimension
                if d == 0:
                    bmin, bmax = bbox[0], bbox[1]
                elif d == 1:
                    bmin, bmax = bbox[2], bbox[3]
                elif d == 2:
                    bmin, bmax = bbox[4], bbox[5]
                # Constraint derived from:
                #   patch_start = center - half  must be <= bmin
                #   patch_end = center - half + s must be >= bmax
                # Thus: center <= bmin + half, and center >= bmax - s + half.
                # Also, the patch must lie within the volume: center in [half, vol_size - s + half]
                center_low = max(bmax - s + half, half)
                center_high = min(bmin + half, vol_size - s + half)
                if center_low > center_high:
                    valid_center = None
                    break
                # 如果是训练模式，则随机选择合法范围内的中心；如果是测试模式，则取中间位置作为中心
                if self.train:
                    valid_center.append(random.randint(center_low, center_high))
                else:
                    valid_center.append((center_low + center_high) // 2)
            if valid_center is None:
                continue
            center = tuple(valid_center)

            # Extract a patch from the original mask using the computed center.
            # Then, binarize: set pixels equal to label_id to 1, and others to 0.
            patch_mask = extract_patch_centered(mask_volume, center, self.target_shape,
                                                pad_mode='constant', constant_values=0)
            patch_mask = np.where(patch_mask == label_id, 1, 0).astype(np.uint8)

            # Extract a patch from the original image volume using the same center.
            if image_volume is not None:
                patch_image = extract_patch_centered(image_volume, center, self.target_shape,
                                                     pad_mode='constant', constant_values=0)
            else:
                patch_image = np.zeros(self.target_shape, dtype=np.int32)

            if self.augment is not None:
                image_tensor_aug = torch.from_numpy(patch_image).unsqueeze(0).float()
                mask_tensor_aug = torch.from_numpy(patch_mask).unsqueeze(0).float()
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image_tensor_aug),
                    mask=tio.LabelMap(tensor=mask_tensor_aug)
                )
                subject = self.augment(subject)
                patch_image = subject.image.data.squeeze(0).numpy()
                patch_mask = subject.mask.data.squeeze(0).numpy()
        
            mask_tensor = torch.from_numpy(patch_mask).float().unsqueeze(0)
            image_tensor = torch.from_numpy(patch_image).float().unsqueeze(0)

            return mask_tensor, image_tensor

        # If maximum retries are exceeded, return zero tensors.
        blank_mask = torch.zeros((1, *self.target_shape), dtype=torch.float32)
        blank_image = torch.zeros((1, *self.target_shape), dtype=torch.float32)
        return blank_mask, blank_image

def main_example_usage():
    from torch.utils.data import DataLoader

    mask_dir = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/train/masks"   # Update with your mask directory
    image_dir = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/train/images" # Update with your image directory

    dataset = InstanceMaskBBoxDataset(
        mask_dir=mask_dir,
        image_dir=image_dir,
        target_shape=(32, 32, 32),
        ignore_label_zero=True,
        min_foreground_pixels=200
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Approximate number of instances:", len(dataset))
    for batch_idx, (mask_tensor, image_tensor) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("  Mask tensor shape:", mask_tensor.shape)
        print("  Image tensor shape:", image_tensor.shape)
        if batch_idx == 0:
            break


class InstanceDataset(Dataset):
    """
    Dataset that loads pre-cropped 3D instance mask TIFF files along with corresponding
    pre-cropped image TIFF files. It assumes that the mask and image files share the same filename.
    """
    def __init__(self, mask_dir, image_dir=None, target_shape=None, augment=None):
        """
        Args:
            mask_dir (str): Directory containing pre-cropped 3D instance mask TIFF files.
            image_dir (str or None): Directory containing pre-cropped 3D image TIFF files.
                If provided, it is assumed that mask and image files share the same filename.
            target_shape (tuple or None): Desired output shape (D, H, W). If provided and the loaded
                volume does not match this shape, you can add a resizing step. For now, it is assumed
                that volumes are already in the desired shape.
        """
        self.mask_dir = Path(mask_dir)
        self.target_shape = target_shape  # (Optional) For future resizing needs.
        self.mask_paths = sorted(self.mask_dir.glob("*.tif"))
        if len(self.mask_paths) == 0:
            raise ValueError(f"No TIFF files found in {self.mask_dir}")

        if image_dir is not None:
            self.image_dir = Path(image_dir)
            self.image_paths = {p.stem: p for p in sorted(self.image_dir.glob("*.tif"))}
        else:
            self.image_dir = None
            self.image_paths = {}
        self.augment = augment

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        # Load pre-cropped instance mask
        mask_path = self.mask_paths[idx]
        mask_volume = load_tiff_as_numpy(mask_path)
        file_stem = mask_path.stem

        # (Optional) Resize the volume to target_shape if needed.
        # For this example, we assume the volumes are already in the desired shape.

        # Load corresponding image patch if available
        if self.image_dir is not None and file_stem in self.image_paths:
            image_volume = load_tiff_as_numpy(self.image_paths[file_stem])
        else:
            # If image not available, create a blank volume with the same shape as the instance.
            image_volume = np.zeros(mask_volume.shape, dtype=mask_volume.dtype)

        if self.augment is not None:
            image_tensor_aug = torch.from_numpy(image_volume).unsqueeze(0).float()
            mask_tensor_aug = torch.from_numpy(mask_volume).unsqueeze(0).float()
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_tensor_aug),
                mask=tio.LabelMap(tensor=mask_tensor_aug)
            )
            subject = self.augment(subject)
            image_volume = subject.image.data.squeeze(0).numpy()
            mask_volume = subject.mask.data.squeeze(0).numpy()
        
        # Convert to torch tensors and add a channel dimension.
        mask_tensor = torch.from_numpy(mask_volume).float().unsqueeze(0)
        image_tensor = torch.from_numpy(image_volume).float().unsqueeze(0)

        return mask_tensor, image_tensor

def get_default_augment():
    return tio.Compose([
        tio.RandomAffine(
            scales=(0.8, 1.2),
            degrees=30,  # 可旋转 ±15°
            translation=(10, 10, 10),  # 平移最多 5 个 voxel
            image_interpolation='linear',
            label_interpolation='nearest',
            p=0.5,
        ),
        tio.RandomNoise(mean=0.0, std=0.1, p=0.5),  # 加入随机噪声
        RandomForegroundMask(block_size=(16,24,24), p=0.5),  # 随机 mask 掉一小块前景
    ])

if __name__ == "__main__":
    main_example_usage()
