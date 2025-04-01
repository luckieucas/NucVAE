import os
import argparse
import logging
from pathlib import Path
import random

import torch
import wandb
import yaml
import tifffile as tiff
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from dotmap import DotMap

# Import your models – note the added dual_vae model.
from models import VAE3D, VQVAE3D, DualEncoderVAE3D  
from dataset_vae import InstanceMaskBBoxDataset, InstanceDataset, get_default_augment


def setup_logging(log_dir):
    """
    Setup logging to file and console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)


def vae_loss_function(x_hat, x, mu, logvar, beta=1.0):
    """
    VAE loss function with beta weighting for KL divergence.
    """
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl


def vqvae_loss_function(x_hat, x, quantized, z_e, beta=0.25):
    """
    Compute loss for VQ-VAE.
    """
    assert quantized.shape == z_e.shape, f"Shape mismatch: quantized {quantized.shape}, z_e {z_e.shape}"
    recon_loss = F.mse_loss(x_hat, x)
    codebook_loss = F.mse_loss(quantized, z_e.detach())
    commitment_loss = F.mse_loss(z_e, quantized.detach())
    total_loss = recon_loss + codebook_loss + beta * commitment_loss
    return total_loss, recon_loss, commitment_loss, codebook_loss


def save_sample_images(inputs, image, recons, epoch, model_type, save_dir, mask_tensor_aug=None):
    """
    Save the first sample from the validation batch along with its reconstruction.
    """
    os.makedirs(save_dir, exist_ok=True)
    sample_input = inputs[0].detach().cpu().numpy()
    sample_recon = recons[0].detach().cpu().numpy()
    sample_img = image[0].detach().cpu().numpy()
    if mask_tensor_aug is not None:
        print(f"Mask tensor aug shape: {mask_tensor_aug.shape}")
        mask_tensor_aug = mask_tensor_aug[0].detach().cpu().numpy()
        mask_aug_filename = os.path.join(save_dir, f"{model_type}_epoch_{epoch}_mask_aug.tif")
        tiff.imwrite(mask_aug_filename, mask_tensor_aug)

    input_filename = os.path.join(save_dir, f"{model_type}_epoch_{epoch}_input.tif")
    recon_filename = os.path.join(save_dir, f"{model_type}_epoch_{epoch}_recon.tif")
    img_filename = os.path.join(save_dir, f"{model_type}_epoch_{epoch}_image.tif")

    tiff.imwrite(img_filename, sample_img)
    tiff.imwrite(input_filename, sample_input)
    tiff.imwrite(recon_filename, sample_recon)
    logging.info(f"Saved validation sample images to {input_filename} and {recon_filename}")


def train_vae(model, dataloader, optimizer, device, beta=1.0):
    """
    Standard training loop for VAE.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(dataloader, start=1):
        data = data.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(data)
        loss, recon_loss, kl = vae_loss_function(x_hat, data, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            wandb.log({
                "batch/loss": loss.item(),
                "batch/recon_loss": recon_loss.item(),
                "batch/kl": kl.item(),
                "batch/beta": beta
            })
            logging.info(f"[Batch {batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} "
                         f"(Recon: {recon_loss.item():.4f}, KL: {kl.item():.4f})")
    return total_loss / len(dataloader)


def train_vqvae(model, dataloader, optimizer, device, beta=0.25):
    """
    Training loop for VQ-VAE.
    """
    model.train()
    total_loss = 0.0
    iterations = 100
    data_iter = iter(dataloader)
    for batch_idx, data in enumerate(dataloader, start=1):
        data = data.to(device)
        optimizer.zero_grad()
        x_hat, quantized, z_e = model(data)
        loss, recon_loss, commitment_loss, codebook_loss = vqvae_loss_function(x_hat, data, quantized, z_e, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            wandb.log({
                "batch/loss": loss.item(),
                "batch/recon_loss": recon_loss.item(),
                "batch/commitment_loss": commitment_loss.item(),
                "batch/codebook_loss": codebook_loss.item()
            })
            logging.info(f"[Batch {batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} "
                         f"(Recon: {recon_loss.item():.4f}, Commitment: {commitment_loss.item():.4f}, "
                         f"Codebook: {codebook_loss.item():.4f})")
    return total_loss / len(dataloader)


def validate_vae(model, dataloader, device, beta=1.0, epoch=None, save_dir=None):
    """
    Validation loop for VAE.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader, start=1):
            data = data.to(device)
            x_hat, mu, logvar = model(data)
            loss, recon_loss, kl = vae_loss_function(x_hat, data, mu, logvar, beta)
            total_loss += loss.item()
            if epoch is not None and save_dir is not None and batch_idx == 1:
                save_sample_images(data, x_hat, epoch, "vae", save_dir)
    avg_loss = total_loss / len(dataloader)
    wandb.log({"epoch/val_loss": avg_loss})
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def validate_vqvae(model, dataloader, device, beta=0.25, epoch=None, save_dir=None):
    """
    Validation loop for VQ-VAE.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader, start=1):
            data = data.to(device)
            x_hat, quantized, z_e = model(data)
            loss, recon_loss, commitment_loss, codebook_loss = vqvae_loss_function(x_hat, data, quantized, z_e, beta)
            total_loss += loss.item()
            if epoch is not None and save_dir is not None and batch_idx == 1:
                save_sample_images(data, x_hat, epoch, "vqvae", save_dir)
    avg_loss = total_loss / len(dataloader)
    wandb.log({"epoch/val_loss": avg_loss})
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def set_requires_grad(modules, requires_grad=False):
    """
    Enable or disable gradients for the given module(s).
    """
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    for m in modules:
        for p in m.parameters():
            p.requires_grad = requires_grad


def train_dual_encoder_phase(model, dataloader, optimizer, device, beta, use_mask_encoder=True):
    """
    Training loop for one phase of the dual encoder VAE.
    
    Args:
        model: Dual encoder VAE model.
        dataloader: DataLoader returning (mask_tensor, image_tensor).
        optimizer: Optimizer.
        device: Device to train on.
        beta: KL divergence weight.
        use_mask_encoder (bool): 
            - True: use mask encoder (input: mask_tensor)
            - False: use image encoder (input: concatenation of image_tensor and mask_tensor)
    
    Returns:
        Average loss over the dataloader.
    """
    model.train()
    total_loss = 0.0
    iterations = 50
    data_iter = iter(dataloader)
    for batch_idx in range(iterations):
        try:
            batch = next(data_iter)
        except StopIteration:
            # If the dataloader is exhausted, restart it
            data_iter = iter(dataloader)
            batch = next(data_iter)
        mask_tensor, image_tensor = batch
        if use_mask_encoder:
            # Phase 1: use only the mask tensor
            input_tensor = mask_tensor.to(device)
            target = mask_tensor.to(device)
        else:
            # Phase 2: create a 2-channel input: [image, mask]
            
            # Normlize image
            image_tensor = image_tensor / 255.0
            # Do dropout augmentation for mask
            mask_tensor_aug = random_block_dropout(mask_tensor, max_dropout=0.8)
            input_tensor = torch.cat([image_tensor, mask_tensor_aug], dim=1).to(device)
            #input_tensor = image_tensor.to(device)
            target = mask_tensor.to(device)  # reconstruction target remains the mask

        optimizer.zero_grad()
        # Forward: use the flag to choose the proper encoder branch
        x_hat, mu, logvar = model(input_tensor, use_mask_encoder=use_mask_encoder)
        loss, recon_loss, kl = vae_loss_function(x_hat, target, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            wandb.log({
                "batch/loss": loss.item(),
                "batch/recon_loss": recon_loss.item(),
                "batch/kl": kl.item(),
                "batch/beta": beta,
                "phase": "mask" if use_mask_encoder else "image"
            })
            logging.info(f"[Batch {batch_idx}/{len(dataloader)}] Phase: {'mask' if use_mask_encoder else 'image'} - Loss: {loss.item():.4f} "
                         f"(Recon: {recon_loss.item():.4f}, KL: {kl.item():.4f})")
    return total_loss / len(dataloader)


def validate_dual_encoder(model, dataloader, device, beta, use_mask_encoder=True, epoch=None, save_dir=None):
    """
    Validation loop for the dual encoder VAE.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (mask_tensor, image_tensor) in enumerate(dataloader, start=1):
            if use_mask_encoder:
                input_tensor = mask_tensor.to(device)
                target = mask_tensor.to(device)
            else:
                # Normalize mask tensor
                image_tensor = image_tensor / 255.0
                mask_tensor_aug = random_block_dropout(mask_tensor, max_dropout=0.8)
                input_tensor = torch.cat([image_tensor, mask_tensor_aug], dim=1).to(device)
                #input_tensor = image_tensor.to(device)
                target = mask_tensor.to(device)
            x_hat, mu, logvar = model(input_tensor, use_mask_encoder=use_mask_encoder)
            loss, recon_loss, kl = vae_loss_function(x_hat, target, mu, logvar, beta)
            total_loss += loss.item()
            if epoch is not None and save_dir is not None and batch_idx == 1:
                save_sample_images(target, image_tensor*255.0, x_hat, 
                                   epoch, "dual_vae", 
                                   save_dir+"_"+"mask_encoder" if use_mask_encoder else save_dir+"_"+"image_encoder",
                                   mask_tensor_aug= None if use_mask_encoder else mask_tensor_aug)
    avg_loss = total_loss / len(dataloader)
    wandb.log({"epoch/val_loss": avg_loss, "phase": "mask" if use_mask_encoder else "image"})
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def random_block_dropout(tensor, block_size=(2, 2, 2), max_dropout=0.5):
    """
    Divide each sample in the 5D tensor (B, C, D, H, W) into blocks of size block_size 
    (applied to the spatial dimensions D, H, W) and randomly set a fraction (0 to max_dropout)
    of those blocks that contain foreground (non-zero values) to zero for each sample.
    
    Args:
      tensor: Input tensor with shape (B, C, D, H, W).
      block_size: Tuple representing the block size in (D, H, W). Default is (2, 2, 2).
      max_dropout: Maximum fraction of candidate blocks (blocks with foreground) to drop (default is 0.6, i.e., 60%).
    
    Returns:
      The augmented tensor with some candidate blocks set to zero for each sample.
    """
    B, C, D, H, W = tensor.shape
    block_D, block_H, block_W = block_size
    
    # Clone the input tensor to avoid modifying the original tensor
    tensor_aug = tensor.clone()
    
    # Process each sample in the batch individually
    for b in range(B):
        sample = tensor_aug[b]  # sample shape: (C, D, H, W)
        
        # Calculate number of blocks in each spatial dimension (using floor division)
        num_blocks_D = D // block_D
        num_blocks_H = H // block_H
        num_blocks_W = W // block_W
        
        # Create a list to hold indices of blocks that contain foreground (non-zero values)
        candidate_blocks = []
        for i in range(num_blocks_D):
            for j in range(num_blocks_H):
                for k in range(num_blocks_W):
                    start_D = i * block_D
                    start_H = j * block_H
                    start_W = k * block_W
                    block_region = sample[:, start_D:start_D+block_D, start_H:start_H+block_H, start_W:start_W+block_W]
                    # Check if this block contains any foreground elements
                    if (block_region > 0).any():
                        candidate_blocks.append((i, j, k))
        
        total_candidate_blocks = len(candidate_blocks)
        if total_candidate_blocks == 0:
            # No candidate blocks with foreground in this sample, skip dropout
            continue
        
        # Randomly decide the fraction of candidate blocks to drop out
        dropout_fraction = random.uniform(0, max_dropout)
        num_dropout = int(total_candidate_blocks * dropout_fraction)
        
        if num_dropout == 0:
            continue
        
        # Randomly select candidate blocks to dropout
        dropout_blocks = random.sample(candidate_blocks, num_dropout)
        
        # Set the selected candidate blocks to zero across all channels
        for i, j, k in dropout_blocks:
            start_D = i * block_D
            start_H = j * block_H
            start_W = k * block_W
            sample[:, start_D:start_D+block_D, start_H:start_H+block_H, start_W:start_W+block_W] = 0
            
    return tensor_aug

def main_train():
    parser = argparse.ArgumentParser()
    # Now support three model types: vae, vqvae, and dual_vae
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("-m", "--model_type", type=str, choices=["vae", "vqvae", "dual_vae"], default="vae")
    args = parser.parse_args()

    # Load configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    # Ensure necessary directories exist
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.join(config.log_dir, "checkpoints"), exist_ok=True)
    val_output_dir = os.path.join(config.log_dir, "val_results")
    os.makedirs(val_output_dir, exist_ok=True)
    
    setup_logging(config.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="3D-VAE-Training",
        config=dict(config),
        name=f"Train_{args.model_type}"
    )
    train_augment = get_default_augment()


    # Prepare datasets
    if args.model_type == "dual_vae":
        train_dataset = InstanceDataset(
            mask_dir=config.data_dir,
            image_dir=config.image_dir,      # provided in config for dual_vae
            target_shape=config.target_shape,
            augment=train_augment
        )
        # train_dataset = InstanceMaskBBoxDataset(
        #     mask_dir=config.data_dir,
        #     image_dir=config.image_dir,      # provided in config for dual_vae
        #     target_shape=config.target_shape,
        #     augment=train_augment
        # )
        val_dataset = InstanceMaskBBoxDataset(
            mask_dir=config.val_dir,
            image_dir=config.val_image_dir,   # provided in config for dual_vae
            target_shape=config.target_shape,
        )
    else:
        train_dataset = InstanceDataset(
            mask_dir=config.data_dir,
            target_shape=config.target_shape,
            augment=train_augment
        )
        val_dataset = InstanceMaskBBoxDataset(
            mask_dir=config.val_dir,
            target_shape=config.target_shape,
            ignore_label_zero=True,
            min_foreground_pixels=config.min_foreground_pixels
        )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model based on type
    if args.model_type == "vae":
        model = VAE3D(
            in_channels=1,
            latent_dim=config.latent_dim,
            base_channel=config.base_channel
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        wandb.watch(model, log_freq=100)
        total_epochs = config.num_epochs
        for epoch in range(total_epochs):
            current_beta = config.beta * (epoch + 1) / total_epochs
            train_loss = train_vae(model, train_loader, optimizer, device, current_beta)
            wandb.log({"epoch/train_loss": train_loss})
            logging.info(f"[Epoch {epoch+1}/{total_epochs}] Train Loss: {train_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                val_loss = validate_vae(model, val_loader, device, current_beta, epoch=epoch+1, save_dir=val_output_dir)
                wandb.log({"epoch/val_loss": val_loss})
                checkpoint_path = os.path.join(config.log_dir, "checkpoints", f"vae_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
        final_model_path = os.path.join(config.log_dir, "vae_final.pth")
        torch.save(model.state_dict(), final_model_path)
        wandb.save(final_model_path)

    elif args.model_type == "vqvae":
        model = VQVAE3D(
            in_channels=1,
            latent_dim=config.latent_dim,
            base_channel=config.base_channel,
            num_embeddings=config.num_embeddings
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        wandb.watch(model, log_freq=100)
        total_epochs = config.num_epochs
        for epoch in range(total_epochs):
            train_loss = train_vqvae(model, train_loader, optimizer, device, beta=0.5)
            wandb.log({"epoch/train_loss": train_loss})
            logging.info(f"[Epoch {epoch+1}/{total_epochs}] Train Loss: {train_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                val_loss = validate_vqvae(model, val_loader, device, beta=0.25, epoch=epoch+1, save_dir=val_output_dir)
                wandb.log({"epoch/val_loss": val_loss})
                checkpoint_path = os.path.join(config.log_dir, "checkpoints", f"vqvae_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
        final_model_path = os.path.join(config.log_dir, "vqvae_final.pth")
        torch.save(model.state_dict(), final_model_path)
        wandb.save(final_model_path)

    elif args.model_type == "dual_vae":
        # Initialize dual encoder VAE
        model = DualEncoderVAE3D(
            mask_in_channels=1,
            img_in_channels=2,
            latent_dim=config.latent_dim,
            base_channel=config.base_channel
        ).to(device)
        wandb.watch(model, log_freq=100)
        # Total training epochs is split into two phases
        phase1_epochs = config.phase1_epochs  # e.g., 50 epochs
        phase2_epochs = config.phase2_epochs  # e.g., 50 epochs

        # ----- Phase 1: Train Mask Encoder and Decoder -----
        # Ensure image encoder is frozen
        set_requires_grad([model.img_enc_conv1, model.img_enc_conv2, model.img_enc_conv3,
                           model.img_enc_mu, model.img_enc_logvar], requires_grad=False)
        # Unfreeze mask encoder and decoder
        set_requires_grad([model.mask_enc_conv1, model.mask_enc_conv2, model.mask_enc_conv3,
                           model.mask_enc_mu, model.mask_enc_logvar,
                           model.dec_conv1, model.dec_conv2, model.dec_conv3], requires_grad=True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        for epoch in range(phase1_epochs):
            train_loss = train_dual_encoder_phase(model, train_loader, optimizer, device, beta=config.beta, use_mask_encoder=True)
            wandb.log({"epoch/train_loss": train_loss, "phase": "mask"})
            logging.info(f"[Phase 1 - Epoch {epoch+1}/{phase1_epochs}] Train Loss: {train_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                val_loss = validate_dual_encoder(model, val_loader, device, beta=config.beta, use_mask_encoder=True, epoch=epoch+1, save_dir=val_output_dir)
                wandb.log({"epoch/val_loss": val_loss, "phase": "mask"})
                checkpoint_path = os.path.join(config.log_dir, "checkpoints", f"dual_vae_mask_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)

        # ----- Phase 2: Freeze Mask Encoder & Decoder, Train Image Encoder -----
        # Freeze mask encoder and decoder
        set_requires_grad([model.mask_enc_conv1, model.mask_enc_conv2, model.mask_enc_conv3,
                           model.mask_enc_mu, model.mask_enc_logvar,
                           model.dec_conv1, model.dec_conv2, model.dec_conv3], requires_grad=False)
        # set_requires_grad([model.mask_enc_conv1, model.mask_enc_conv2, model.mask_enc_conv3,
        #                    model.mask_enc_mu, model.mask_enc_logvar,
        #                    model.dec_conv1, model.dec_conv2, model.dec_conv3], requires_grad=True)
        
        # Unfreeze image encoder
        set_requires_grad([model.img_enc_conv1, model.img_enc_conv2, model.img_enc_conv3,
                           model.img_enc_mu, model.img_enc_logvar], requires_grad=True)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        for epoch in range(phase2_epochs):
            train_loss = train_dual_encoder_phase(model, train_loader, optimizer, device, beta=config.beta, use_mask_encoder=False)
            wandb.log({"epoch/train_loss": train_loss, "phase": "image"})
            logging.info(f"[Phase 2 - Epoch {epoch+1}/{phase2_epochs}] Train Loss: {train_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                val_loss = validate_dual_encoder(model, val_loader, device, beta=config.beta, use_mask_encoder=False, epoch=phase1_epochs+epoch+1, save_dir=val_output_dir)
                wandb.log({"epoch/val_loss": val_loss, "phase": "image"})
                checkpoint_path = os.path.join(config.log_dir, "checkpoints", f"dual_vae_image_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
                
        final_model_path = os.path.join(config.log_dir, "dual_vae_final.pth")
        torch.save(model.state_dict(), final_model_path)
        wandb.save(final_model_path)

if __name__ == "__main__":
    main_train()
