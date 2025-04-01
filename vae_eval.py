import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import numpy as np
from models import VAE3D, DualEncoderVAE3D # For standard VAE; dual encoder model is imported later if needed



def load_tiff_as_numpy(filepath):
    """Load 3D TIFF file as a NumPy array"""
    return tifffile.imread(filepath).astype(np.int32)

def compute_log_likelihood(vae_model, x, num_samples=100):
    """
    Estimate log-likelihood of a single 3D instance using importance sampling
    Args:
        vae_model: Pretrained VAE model
        x (torch.Tensor): Input tensor of shape (1, D, H, W)
        num_samples (int): Number of Monte Carlo samples
    Returns:
        float: Estimated log-likelihood
    """
    vae_model.eval()
    with torch.no_grad():
        # Add batch and channel dimensions -> (1,1,D,H,W)
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
                        help="Folder containing 32x32x32 instance TIFF files")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained VAE model")
    parser.add_argument("--output_txt", type=str, required=True,
                        help="Path to save the log-likelihood results as a txt file")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of Monte Carlo samples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference, e.g., cuda or cpu")
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="Latent dimension of the VAE model")
    parser.add_argument("--base_channel", type=int, default=16,
                        help="Base number of channels in the VAE model")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    model = DualEncoderVAE3D(
            mask_in_channels=1,
            img_in_channels=2,
            latent_dim=args.latent_dim,
            base_channel=args.base_channel
        ).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # Get all instance files
    instance_files = list(input_folder.glob("*.tif")) + list(input_folder.glob("*.tiff"))
    if not instance_files:
        print("No instance files found in the input folder.")
        return

    # Compute log-likelihood for each instance, store as (filename, loglikelihood) pair
    results = []
    for file in instance_files:
        instance = load_tiff_as_numpy(file)
        # Assume each instance is 32x32x32, convert to tensor of shape (1, D, H, W)
        instance_tensor = torch.from_numpy(instance).type(torch.float32).unsqueeze(0).to(device)
        print(f"instance_tensor shape: {instance_tensor.shape}")
        ll = compute_log_likelihood(model, instance_tensor, num_samples=args.num_samples)
        results.append((file.name, ll))
        print(f"Processing {file.name}: log-likelihood = {ll:.4f}")

    # Sort results by log-likelihood in ascending order
    results.sort(key=lambda x: x[1])

    # Write results to txt file
    with open(args.output_txt, "w") as out_file:
        for filename, ll in results:
            out_file.write(f"{filename}\t{ll:.4f}\n")
    print("All results have been sorted by log-likelihood and saved to", args.output_txt)

if __name__ == "__main__":
    main()
