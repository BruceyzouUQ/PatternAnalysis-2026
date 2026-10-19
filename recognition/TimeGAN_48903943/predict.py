from dataset import get_datasets
from modules import TimeVAE
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader

DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
train_set, val_set, test_set, norm = get_datasets(DATA_PATH)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
output_dir = os.path.expanduser("~/baselineOutputs40")
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained model ---
model = TimeVAE(latent_dim=128).to(device)
model.load_state_dict(torch.load(
    os.path.join(output_dir, "timevae40_best.pt"),
    map_location=device))
model.eval()

def to_depth_heatmap(Z, mid, use_log=True):
    # Z: (64, 40) normalised window, mid: (64,)
    real = norm.inverse(Z, mid)
    ask_v = real[:, 1::4]
    bid_v = real[:, 3::4]
    img = np.vstack((ask_v.T[::-1], bid_v.T))
    return np.log1p(img) if use_log else img



def visualise_reconstructions(model, val_loader, n=8):
    x, mid = next(iter(val_loader))
    with torch.no_grad():
        x_hat, mu, logvar = model(x.to(device))
    x, x_hat = x.numpy(), x_hat.cpu().numpy()
    mid = mid.numpy()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        orig = to_depth_heatmap(x[i], mid[i])
        recon = to_depth_heatmap(x_hat[i], mid[i])
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

def visualise_samples(model, val_set, n=8):
    samples = model.sample(n).cpu().numpy()          # (n, 64, 40)
    mid = val_set[0][1].numpy()                      # any real (64,) midpoint; volumes ignore it

    imgs = [to_depth_heatmap(samples[i], mid) for i in range(n)]
    vmax = max(img.max() for img in imgs)            # one shared colour scale

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    for i, img in enumerate(imgs):
        axes[i].imshow(img, aspect="auto", vmin=0, vmax=vmax)
        axes[i].axis("off")
    axes[0].set_title("Random Samples", loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "randsamples.png"))
    plt.close()

if __name__ == "__main__":
    visualise_reconstructions(model, val_loader)
    visualise_samples(model, val_set)