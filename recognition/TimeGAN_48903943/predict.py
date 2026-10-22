from dataset import get_datasets, LOBNormaliser
from modules import TimeVAE, TimeGAN
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader

def to_depth_heatmap(Z, mid, norm, use_log=True):
    real = norm.inverse(Z, mid)
    ask_v = real[:, 1::4]
    bid_v = real[:, 3::4]
    img = np.vstack((ask_v.T[::-1], bid_v.T))
    return np.log1p(img) if use_log else img

def visualise_reconstructions(model, val_loader, norm, device, output_dir, n=8):
    x, mid = next(iter(val_loader))
    with torch.no_grad():
        x_hat, mu, logvar = model(x.to(device))
    x, x_hat = x.numpy(), x_hat.cpu().numpy()
    mid = mid.numpy()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        orig = to_depth_heatmap(x[i], mid[i], norm)
        recon = to_depth_heatmap(x_hat[i], mid[i], norm)
        vmax = max(orig.max(), recon.max())      # shared colour scale per pair
        axes[0, i].imshow(orig, aspect="auto", vmin=0, vmax=vmax)
        axes[1, i].imshow(recon, aspect="auto", vmin=0, vmax=vmax)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Reconstructed", loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstructions.png"))
    plt.close()

def visualise_samples(model, val_set, norm, device, output_dir, model_type, n=8):
    with torch.no_grad():
        if model_type == "vae":
            samples = model.sample(n)
        else:
            z = torch.randn(n, 64, model.noise_dim, device=device)
            samples = model.generate(z)

    samples = samples.cpu().numpy()

    mid = val_set[0][1].numpy()

    imgs = [
        to_depth_heatmap(samples[i], mid, norm)
        for i in range(n)
    ]

    real_samples = [
        val_set[i][0].numpy()
        for i in range(n)
    ]

    real_imgs = [
        to_depth_heatmap(real_samples[i], mid, norm)
        for i in range(n)
    ]

    vmax = max(
        max(img.max() for img in real_imgs),
        max(img.max() for img in imgs)
    )

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))

    for i in range(n):
        axes[0, i].imshow(real_imgs[i], aspect="auto", vmin=0, vmax=vmax)
        axes[1, i].imshow(imgs[i], aspect="auto", vmin=0, vmax=vmax)

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Real", loc="left")
    axes[1, 0].set_title("TimeGAN", loc="left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_vs_timegan.png"))
    plt.close()

def main():
    model_type = "timegan"

    DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    train_set, val_set, test_set, _ = get_datasets(DATA_PATH)

    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "vae":
        output_dir = os.path.expanduser("~/baselineOutputs")
        model = TimeVAE(latent_dim=128).to(device)
        checkpoint = "timevae_best.pt"

    elif model_type == "timegan":
        output_dir = os.path.expanduser("~/TimeGANoutputs")
        model = TimeGAN().to(device)
        checkpoint = "timegan_best.pt"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, checkpoint),
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    norm = LOBNormaliser.load(
        os.path.join(output_dir, "normaliser.joblib")
    )

    if model_type == "vae":
        visualise_reconstructions(model, val_loader, norm, device, output_dir)

    visualise_samples(model, val_set, norm, device, output_dir, model_type)


if __name__ == "__main__":
    main()