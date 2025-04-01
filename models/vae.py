import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE3D(nn.Module):
    def __init__(self, 
                 in_channels=1,   # Number of input channels (e.g., 1 for single-channel masks)
                 latent_dim=64,   # Number of channels in the latent space
                 base_channel=16  # Base number of channels for the first convolutional layer
                 ):
        super(VAE3D, self).__init__()
        
        # ---------- Encoder ----------
        self.enc_conv1 = nn.Conv3d(in_channels, base_channel, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv3d(base_channel, base_channel*2, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv3d(base_channel*2, base_channel*4, kernel_size=3, stride=2, padding=1)
        
        # Using 1x1x1 convolutions to obtain latent space parameters (mu, logvar)
        self.conv_mu = nn.Conv3d(base_channel*4, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv3d(base_channel*4, latent_dim, kernel_size=1)
        
        # ---------- Decoder ----------
        # Decoder uses upsampling + convolution to reconstruct the input shape dynamically
        self.dec_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(latent_dim, base_channel*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(base_channel*2, base_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(base_channel, 1, kernel_size=3, padding=1)
        )
        
    def encode(self, x):
        """Encodes input into latent space representation (mu, logvar)"""
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        mu = self.conv_mu(x)      # Compute mean
        logvar = self.conv_logvar(x)  # Compute log-variance
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from latent space"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decodes latent space representation back to original input space"""
        x = self.dec_conv1(z)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)
        x = torch.sigmoid(x)  # Ensure output values are between (0,1)
        return x
    
    def forward(self, x):
        """Full forward pass from input to reconstructed output"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


class VQVAE3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, base_channel=16, num_embeddings=512):
        super(VQVAE3D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channel, base_channel*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channel*2, latent_dim, kernel_size=4, stride=2, padding=1)
        )

        # Vector Quantization
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = latent_dim

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channel*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(base_channel, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        z_e = self.encoder(x)  # (B, latent_dim, D, H, W)
        
        B, C, D, H, W = z_e.shape
        z_e_flattened = z_e.view(B, C, -1).permute(0, 2, 1)  # (B, D*H*W, C)
        
        distances = torch.cdist(z_e_flattened, self.codebook.weight)  # (B, D*H*W, num_embeddings)
        encoding_indices = torch.argmin(distances, dim=-1)  # (B, D*H*W)
        
        quantized = self.codebook(encoding_indices).permute(0, 2, 1).view(B, C, D, H, W)
        
        # Apply the straight-through estimator
        quantized = z_e + (quantized - z_e).detach()
        
        return z_e, quantized, encoding_indices

    def forward(self, x):
        """VQVAE Forward Pass."""
        z_e, quantized, encoding_indices = self.encode(x)
        x_hat = self.decoder(quantized)
        return x_hat, quantized, z_e



class DualEncoderVAE3D(nn.Module):
    def __init__(self, 
                 mask_in_channels=1,   # Input channels for mask_encoder (1 for mask)
                 img_in_channels=2,    # Input channels for image_encoder (2: image + mask)
                 latent_dim=64,        # Dimension of the latent space
                 base_channel=16):     # Base number of channels
        super(DualEncoderVAE3D, self).__init__()
        
        # -------------------- Mask Encoder --------------------
        self.mask_enc_conv1 = nn.Conv3d(mask_in_channels, base_channel, kernel_size=3, stride=2, padding=1)
        self.mask_enc_conv2 = nn.Conv3d(base_channel, base_channel*2, kernel_size=3, stride=2, padding=1)
        self.mask_enc_conv3 = nn.Conv3d(base_channel*2, base_channel*4, kernel_size=3, stride=2, padding=1)

        self.mask_enc_mu     = nn.Conv3d(base_channel*4, latent_dim, kernel_size=1)
        self.mask_enc_logvar = nn.Conv3d(base_channel*4, latent_dim, kernel_size=1)
        
        # -------------------- Image Encoder --------------------
        self.img_enc_conv1 = nn.Conv3d(img_in_channels, base_channel, kernel_size=3, stride=2, padding=1)
        self.img_enc_conv2 = nn.Conv3d(base_channel, base_channel*2, kernel_size=3, stride=2, padding=1)
        self.img_enc_conv3 = nn.Conv3d(base_channel*2, base_channel*4, kernel_size=3, stride=2, padding=1)
        #self.img_enc_bn1 = nn.BatchNorm3d(base_channel)
        #self.img_enc_bn2 = nn.BatchNorm3d(base_channel*2)
        #self.img_enc_bn3 = nn.BatchNorm3d(base_channel*4)
        self.img_enc_mu     = nn.Conv3d(base_channel*4, latent_dim, kernel_size=1)
        self.img_enc_logvar = nn.Conv3d(base_channel*4, latent_dim, kernel_size=1)
        
        # -------------------- Shared Decoder --------------------
        self.dec_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(latent_dim, base_channel*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(base_channel*2, base_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(base_channel, 1, kernel_size=3, padding=1)
        )

    def encode_mask(self, x_mask):
        """Forward pass through the mask encoder"""
        x_mask = F.relu(self.mask_enc_conv1(x_mask))
        x_mask = F.relu(self.mask_enc_conv2(x_mask))
        x_mask = F.relu(self.mask_enc_conv3(x_mask))
        mu     = self.mask_enc_mu(x_mask)
        logvar = self.mask_enc_logvar(x_mask)
        return mu, logvar

    def encode_image(self, x_img):
        """Forward pass through the image encoder"""
        x_img = F.relu(self.img_enc_conv1(x_img))
        x_img = F.relu(self.img_enc_conv2(x_img))
        x_img = F.relu(self.img_enc_conv3(x_img))
        mu     = self.img_enc_mu(x_img)
        logvar = self.img_enc_logvar(x_img)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode the latent vector z into output space"""
        x = self.dec_conv1(z)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)
        x = torch.sigmoid(x)  # Ensure output values are in (0,1)
        return x

    def forward(self, x, use_mask_encoder=True):
        """
        Forward pass using the selected encoder:
          - use_mask_encoder = True  → Use mask_encoder (x: [B, 1, D, H, W])
          - use_mask_encoder = False → Use image_encoder (x: [B, 2, D, H, W])
        """
        if use_mask_encoder:
            mu, logvar = self.encode_mask(x)
        else:
            mu, logvar = self.encode_image(x)
        
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        
        return x_hat, mu, logvar