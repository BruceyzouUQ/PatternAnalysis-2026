import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # 5 conv blocks, each stride=2, doubling channels roughly:
        # each block: Conv1d + BatchNorm1d + activation
        # input: (B, 40, 64) -> output: (B, 512, 2)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        flattened_dim = 512 * 2
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1) # (B, 1024)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# x = torch.randn(8, 64, 40)
# mu, logvar = Encoder()(x.permute(0, 2, 1))
# print(mu.shape, logvar.shape)   # expect torch.Size([8, 128]) each

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        flattened_dim = 512 * 2
        self.fc = nn.Linear(latent_dim, flattened_dim)

        # mirror of encoder: 5 ConvTranspose1d blocks
        # (B, 512, 2) -> (B, 40, 64)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=40, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 2)  # unflatten
        x = self.deconv_layers(x)
        return x

z = torch.randn(8, 128)
out = Decoder()(z)
print(out.shape)   # want (8, 40, 64)

# Test encoder + decoder
x = torch.randn(8, 64, 40).permute(0, 2, 1)
mu, logvar = Encoder()(x)
print(Decoder()(mu).shape == x.shape)