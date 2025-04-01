import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import numpy as np
from models import VAE3D, DualEncoderVAE3D # For standard VAE; dual encoder model is imported later if needed



def load_tiff_as_numpy(filepath):
    """加载 3D TIFF 文件为 NumPy 数组"""
    return tifffile.imread(filepath).astype(np.int32)

def compute_log_likelihood(vae_model, x, num_samples=100):
    """
    使用重要性采样估计单个 3D 实例的 log-likelihood
    Args:
        vae_model: 预训练的 VAE 模型
        x (torch.Tensor): 输入张量，形状为 (1, D, H, W)
        num_samples (int): Monte Carlo 采样次数
    Returns:
        float: 估计的 log-likelihood
    """
    vae_model.eval()
    with torch.no_grad():
        # 增加 batch 和 channel 维度，变为 (1,1,D,H,W)
        x = x.unsqueeze(0)
        mu, logvar = vae_model.encode_mask(x)
        z_samples = []
        for _ in range(num_samples):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            z_samples.append(z)
        z_samples = torch.stack(z_samples, dim=0)
        log_p_x_given_z = []
        for z in z_samples:
            x_hat = vae_model.decode(z)
            likelihood = F.binary_cross_entropy(x_hat, x, reduction='sum')
            log_p_x_given_z.append(-likelihood)
        log_p_x_given_z = torch.stack(log_p_x_given_z)
        log_likelihood = torch.logsumexp(log_p_x_given_z, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32))
    return log_likelihood.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="包含已裁剪为 32x32x32 的实例 TIFF 文件的文件夹")
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练 VAE 模型的路径")
    parser.add_argument("--output_txt", type=str, required=True,
                        help="输出 log-likelihood 结果的 txt 文件路径")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Monte Carlo 采样次数")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理使用的设备，例如 cuda 或 cpu")
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="VAE 模型的 latent 维度")
    parser.add_argument("--base_channel", type=int, default=16,
                        help="VAE 模型的基础通道数")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载预训练模型
    model = DualEncoderVAE3D(
            mask_in_channels=1,
            img_in_channels=2,
            latent_dim=args.latent_dim,
            base_channel=args.base_channel
        ).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("模型加载成功。")

    # 获取所有实例文件
    instance_files = list(input_folder.glob("*.tif")) + list(input_folder.glob("*.tiff"))
    if not instance_files:
        print("未在输入文件夹中找到实例文件。")
        return

    # 计算每个实例的 log-likelihood，存储为 (filename, loglikelihood) 对
    results = []
    for file in instance_files:
        instance = load_tiff_as_numpy(file)
        # 假设每个实例尺寸为 32x32x32，转换为 tensor，形状为 (1, D, H, W)
        instance_tensor = torch.from_numpy(instance).type(torch.float32).unsqueeze(0).to(device)
        print(f"instance_tensor shape: {instance_tensor.shape}")
        ll = compute_log_likelihood(model, instance_tensor, num_samples=args.num_samples)
        results.append((file.name, ll))
        print(f"处理 {file.name}：log-likelihood = {ll:.4f}")

    # 按 log-likelihood 从小到大排序
    results.sort(key=lambda x: x[1])

    # 将结果写入 txt 文件
    with open(args.output_txt, "w") as out_file:
        for filename, ll in results:
            out_file.write(f"{filename}\t{ll:.4f}\n")
    print("所有结果已按 log-likelihood 从小到大排列，并保存至", args.output_txt)

if __name__ == "__main__":
    main()
