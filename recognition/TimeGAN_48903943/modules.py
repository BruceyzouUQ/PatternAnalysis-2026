import torch
import torch.nn as nn
import torch.nn.functional as F

#---------------------------------TIMEVAE----------------------------------
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

# z = torch.randn(8, 128)
# out = Decoder()(z)
# print(out.shape)   # want (8, 40, 64)

# Test encoder + decoder
# x = torch.randn(8, 64, 40).permute(0, 2, 1)
# mu, logvar = Encoder()(x)
# print(Decoder()(mu).shape == x.shape)


class TimeVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):                      # x: (B, 64, 40)
        mu, logvar = self.encoder(x.permute(0, 2, 1))
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z).permute(0, 2, 1)   # back to (B, 64, 40)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return self.decoder(z).permute(0, 2, 1)


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    b = x.size(0)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / b
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ---------------------------------TIMEGAN--------------------------------

class Embedder(nn.Module):
    def __init__(self, feature_dim=40, latent_dim=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            feature_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.projection(h)


class Recovery(nn.Module):
    def __init__(self, feature_dim=40, latent_dim=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        self.projection = nn.Linear(latent_dim, feature_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.projection(h)

# embedder = Embedder()
# recovery = Recovery()
#
# optimizer = torch.optim.Adam(
#     list(embedder.parameters()) + list(recovery.parameters()),
#     lr=1e-3
# )
#
# x = torch.randn(8, 64, 40)
#
# optimizer.zero_grad()
#
# h = embedder(x)
# x_hat = recovery(h)
#
# loss = nn.functional.mse_loss(x_hat, x)
#
# loss.backward()
# optimizer.step()
#
# print(loss.item())


class Supervisor(nn.Module):
    def __init__(self, latent_dim=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.projection(out)

# x = torch.randn(8, 64, 40)
#
# embedder = Embedder()
# supervisor = Supervisor()
#
# h = embedder(x)
# h_hat = supervisor(h)
#
# print(h.shape)      # (8, 64, 128)
# print(h_hat.shape)  # (8, 64, 128)

class Generator(nn.Module):
    def __init__(self, noise_dim=40, latent_dim=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            noise_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        return self.projection(h)

# z = torch.randn(8, 64, 40)
#
# generator = Generator()
# supervisor = Supervisor()
# recovery = Recovery()
#
# h = generator(z)
# h_hat = supervisor(h)
# x_hat = recovery(h_hat)
#
# print(h.shape)
# print(h_hat.shape)
# print(x_hat.shape)

class Discriminator(nn.Module):
    def __init__(self, latent_dim=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        self.projection = nn.Linear(latent_dim, 1)

    def forward(self, h):
        out, _ = self.gru(h)
        return self.projection(out)

z = torch.randn(8, 64, 40)

generator = Generator()
supervisor = Supervisor()
discriminator = Discriminator()

h = generator(z)
h_hat = supervisor(h)
scores = discriminator(h_hat)

print(h.shape)       # (8, 64, 128)
print(h_hat.shape)   # (8, 64, 128)
print(scores.shape)  # (8, 64, 1)

class TimeGAN(nn.Module):
    pass

