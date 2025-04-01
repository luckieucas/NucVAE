import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, base_channel=16, num_embeddings=512):
        super(VQVAE3D, self).__init__()

        # Encoder (Fully Convolutional)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channel, base_channel*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channel*2, latent_dim, kernel_size=4, stride=2, padding=1)  # Output: BxCxdxhxw
        )

        # Vector Quantization
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = latent_dim

        # Decoder (Fully Convolutional)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channel*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(base_channel, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input x into a latent discrete code."""
        z = self.encoder(x)  # Output: BxCxdxhxw
        z_flattened = z.view(z.shape[0], self.embedding_dim, -1).permute(0, 2, 1)  # Reshape
        distances = torch.cdist(z_flattened, self.codebook.weight)  # Compute distances
        encoding_indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook(encoding_indices).permute(0, 2, 1).view_as(z)  # Reshape back
        return quantized, encoding_indices

    def forward(self, x):
        quantized, encoding_indices = self.encode(x)
        x_hat = self.decoder(quantized)  # Output size depends on input
        return x_hat, quantized, encoding_indices
