import numpy as np
from scipy.stats import kurtosis
from skimage.metrics import structural_similarity
from dataset import get_datasets, LOBNormaliser
from modules import TimeVAE, TimeGAN
import torch
import os

# Model Sampling
def generate_samples(model, model_type, n, device, batch_size=64):
    samples = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)

            if model_type == "vae":
                x = model.sample(b)
            else:
                z = torch.randn(b, 64, model.noise_dim, device=device)
                x = model.generate(z)

            samples.append(x.cpu().numpy())

    return np.concatenate(samples, axis=0)

def main():
    model_type = "timegan"  # "vae" or "timegan"

    DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    train_set, val_set, test_set, _ = get_datasets(DATA_PATH)

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

if __name__ == "__main__":
    main()