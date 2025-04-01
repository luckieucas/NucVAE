import argparse
import torch
import numpy as np
import torch.nn.functional as F
import tifffile  # Used to save TIFF files
from models import VAE3D, DualEncoderVAE3D # For standard VAE; dual encoder model is imported later if needed

def compute_log_likelihood(vae_model, x, num_samples=100, use_mask_encoder=True):
    """
    Estimate the log-likelihood of a new sample x using Importance Sampling.
    
    Args:
        vae_model: Trained VAE model.
        x (torch.Tensor): Input data (e.g., (1, D, H, W) or (2, D, H, W) for dual encoder).
        num_samples (int): Number of Monte Carlo samples.
        use_mask_encoder (bool): For dual encoder VAE, use mask branch if True.
    
    Returns:
        float: Estimated log-likelihood.
    """
    vae_model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0)  # (1, C, D, H, W)
        try:
            _, mu, logvar = vae_model(x, use_mask_encoder=use_mask_encoder)
        except TypeError:
            mu, logvar = vae_model.encode(x)
        z_samples = []
        for _ in range(num_samples):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            z_samples.append(z)
        z_samples = torch.stack(z_samples, dim=0)  # (num_samples, batch, latent_dim)
        
        log_p_x_given_z = []
        for z in z_samples:
            x_hat = vae_model.decode(z)
            likelihood = F.binary_cross_entropy(x_hat, x, reduction='sum')
            log_p_x_given_z.append(-likelihood)
        log_p_x_given_z = torch.stack(log_p_x_given_z)  # (num_samples, batch)
        
        log_q_z_given_x = -0.5 * torch.sum(logvar + torch.exp(logvar) + mu**2 - 1, dim=1)
        log_p_z = -0.5 * torch.sum(z_samples ** 2, dim=2)
        log_weights = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32))
    return log_likelihood.mean().item()


def compute_elbo(vae_model, x, use_mask_encoder=True):
    """
    Compute the Evidence Lower Bound (ELBO) as a lower bound on log-likelihood.
    
    Args:
        vae_model: Trained VAE model.
        x (torch.Tensor): Input data.
        use_mask_encoder (bool): For dual encoder VAE, use mask branch if True.
    
    Returns:
        float: ELBO value.
    """
    vae_model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0)
        try:
            x_hat, mu, logvar = vae_model(x, use_mask_encoder=use_mask_encoder)
        except TypeError:
            x_hat, mu, logvar = vae_model(x)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)
        elbo = -(recon_loss + kl_div)
    return elbo.item()


def vae_inference(vae_model, x, device, use_mask_encoder=True):
    """
    Perform inference on a single 3D sample x.
    For dual encoder VAE, select the branch via use_mask_encoder.
    
    Args:
        vae_model: Trained VAE model.
        x (torch.Tensor): Input sample (e.g., (1, D, H, W) or (2, D, H, W)).
        device: Device to run inference.
        use_mask_encoder (bool): For dual encoder, True uses mask branch.
    
    Returns:
        tuple: (reconstruction error, KL divergence, reconstructed output x_hat).
    """
    vae_model.eval()
    with torch.no_grad():
        x = x.to(device).unsqueeze(0)  # (1, C, D, H, W)
        try:
            x_hat, mu, logvar = vae_model(x, use_mask_encoder=use_mask_encoder)
        except TypeError:
            x_hat, mu, logvar = vae_model(x)
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)
    return kl.item(), x_hat


def evaluate_new_data(vae_model, new_data, use_mask_encoder=True):
    """
    Evaluate new data by computing log-likelihood and ELBO.
    
    Args:
        vae_model: Trained VAE model.
        new_data (torch.Tensor): New input sample.
        use_mask_encoder (bool): For dual encoder, select branch.
    
    Returns:
        tuple: (log-likelihood, ELBO)
    """
    log_likelihood = compute_log_likelihood(vae_model, new_data, num_samples=100, use_mask_encoder=use_mask_encoder)
    elbo = compute_elbo(vae_model, new_data, use_mask_encoder=use_mask_encoder)
    print(f"Log-Likelihood Estimate: {log_likelihood:.4f}")
    print(f"ELBO (Lower Bound): {elbo:.4f}")
    return log_likelihood, elbo


def main_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_files", type=str, nargs="+", required=True,
                        help="List of TIFF mask files to process")
    parser.add_argument("--image_files", type=str, nargs="+", default=None,
                        help="List of TIFF image files corresponding to the mask files (for dual encoder)")
    parser.add_argument("--model_type", type=str, choices=["vae", "dual_vae"], default="vae",
                        help="Type of model to test")
    parser.add_argument("--model_path", type=str, default="./models_dir/vae3d_epoch_10000.pth",
                        help="Path to model weights")
    parser.add_argument("--encoder2_channels", type=int, default=2,)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model weights based on model type
    if args.model_type == "vae":
        model = VAE3D(in_channels=1, latent_dim=16, base_channel=16).to(device)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Standard VAE model weights loaded successfully.")
    elif args.model_type == "dual_vae":
        model = DualEncoderVAE3D(
            mask_in_channels=1,
            img_in_channels=args.encoder2_channels,
            latent_dim=16,
            base_channel=16
        ).to(device)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Dual Encoder VAE model weights loaded successfully.")
    
    # Process each test mask file.
    # If --image_files is provided, they are assumed to correspond (in order) to the mask files.
    for idx, test_file in enumerate(args.process_files):
        print(f"\nProcessing mask file: {test_file}")
        # Load mask (assumed to be a 3D patch of shape (32, 32, 32))
        mask_arr = tifffile.imread(test_file)
        mask_tensor = torch.from_numpy(mask_arr)
        mask_tensor = mask_tensor.unsqueeze(0).type(torch.float32)
        
        if args.model_type == "dual_vae":
            print(f"Testing dual encoder model on {test_file}")
            # Test mask encoder branch using the mask alone
            kl_mask, x_hat_mask = vae_inference(model, mask_tensor, device, use_mask_encoder=True)
            #evaluate_new_data(model, mask_tensor.to(device), use_mask_encoder=True)
            
            # For the image branch, try to load corresponding image if provided
            if args.image_files is not None and idx < len(args.image_files):
                image_file = args.image_files[idx]
                print(f"Using corresponding image file: {image_file}")
                image_arr = tifffile.imread(image_file)
                image_tensor = torch.from_numpy(image_arr)
            else:
                print("No corresponding image file provided; using mask as fallback.")
                image_tensor = None
            image_tensor = image_tensor.unsqueeze(0).type(torch.float32) 
            image_tensor = image_tensor / 255.0
            if image_tensor is not None:
                # Form a two-channel input: first channel = image, second channel = mask
                image_input = torch.cat([image_tensor,mask_tensor], dim=0)
            else:
                # Fallback: simulate two-channel input by stacking the mask twice
                image_input = torch.stack([mask_tensor, mask_tensor], dim=0)
            
            kl_img, x_hat_img = vae_inference(model, image_input, device, use_mask_encoder=False)
            print(f"Image Encoder Branch: kl={kl_img:.4f}")
            #evaluate_new_data(model, image_input.to(device), use_mask_encoder=False)
            
            # Save reconstructed outputs for both branches
            x_hat_mask_np = x_hat_mask.cpu().numpy()[0, 0]  # remove batch and channel dims
            x_hat_img_np = x_hat_img.cpu().numpy()[0, 0]
            tifffile.imwrite(test_file.replace(".tif", "_reconstructed_mask.tif"), x_hat_mask_np)
            tifffile.imwrite(test_file.replace(".tif", "_reconstructed_image.tif"), x_hat_img_np)
            print("Reconstructed outputs for both branches saved.")
        else:
            print(f"Testing standard VAE model on {test_file}")
            kl_val, x_hat = vae_inference(model, mask_tensor, device)
            print(f"Inference for {test_file}: kl={kl_val:.4f}")
            #evaluate_new_data(model, mask_tensor.to(device))
            x_hat_np = x_hat.cpu().numpy()[0, 0]
            tifffile.imwrite(test_file.replace(".tif", "_reconstructed.tif"), x_hat_np)
            print("Reconstructed result saved.")

if __name__ == "__main__":
    main_test()
