import numpy as np
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

# Financial validity
def rule_violations(Z, norm, eps=1e-6):
    Z = np.asarray(Z, dtype=np.float64).reshape(-1, 40)

    gap = np.exp(norm.p_scaler.inverse_transform(Z[:, :18]))        # ticks
    vol = np.expm1(norm.v_scaler.inverse_transform(Z[:, 18:38]))    # shares
    spr = np.exp(norm.s_scaler.inverse_transform(Z[:, 38:39]).ravel())

    return {
        "spread_below_one_tick":    np.mean(spr < 1.0 - eps),
        "locked_or_crossed":        np.mean(spr <= 0),
        "ask_gap_below_one_tick":   np.mean((gap[:, :9] < 1.0 - eps).any(axis=1)),
        "bid_gap_below_one_tick":   np.mean((gap[:, 9:] < 1.0 - eps).any(axis=1)),
        "ask_ladder":               np.mean((gap[:, :9] <= 0).any(axis=1)),
        "bid_ladder":               np.mean((gap[:, 9:] <= 0).any(axis=1)),
        "neg_volume":               np.mean((vol < -eps).any(axis=1)),
    }

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

    gen_Z = generate_samples(model, model_type, 2000, device)
    real_Z = test_set.Z.numpy()

    print("Generated:", gen_Z.shape)
    print("Real:", real_Z.shape)

    print("Real violations:", rule_violations(real_Z, norm))
    print("Generated violations:", rule_violations(gen_Z, norm))

if __name__ == "__main__":
    main()