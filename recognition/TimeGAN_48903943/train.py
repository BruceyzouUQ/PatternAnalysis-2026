import os

import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import get_datasets
from modules import TimeVAE, vae_loss, TimeGAN


def run_epoch(model, loader, device, beta, optimizer=None):
    """One pass over loader; returns per-window mean [total, recon, kl]."""
    training = optimizer is not None
    model.train(training)                 # BatchNorm: batch stats vs running stats
    sums, n = torch.zeros(3), 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = x.size(0)
            sums += torch.tensor([loss.item(), recon.item(), kl.item()]) * b
            n += b
    return (sums / n).tolist()


def vae():
    num_epochs, beta, lr = 100, 1.0, 1e-3

    # local
    # train_set, val_set, test_set, norm = get_datasets("order_sample.csv")

    # rangpur
    DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    train_set, val_set, test_set, norm = get_datasets(DATA_PATH)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeVAE(latent_dim=128).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    checkpoint_dir = os.path.expanduser("~/baselineOutputs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    norm.save(os.path.join(checkpoint_dir, "normaliser.joblib"))        # fitted on train only; predict.py reloads it

    history = {"train": [], "val": []}
    best_val = float("inf")
    for epoch in range(num_epochs):
        tr = run_epoch(model, train_loader, device, beta, optimizer)
        va = run_epoch(model, val_loader, device, beta)
        history["train"].append(tr)
        history["val"].append(va)
        if va[0] < best_val:
            best_val = va[0]
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "timevae_best.pt"))
        print(f"Epoch {epoch+1}/{num_epochs} | train {tr[0]:.1f} "
              f"(recon {tr[1]:.1f}, kl {tr[2]:.1f}) | val {va[0]:.1f} "
              f"(recon {va[1]:.1f}, kl {va[2]:.1f})")

    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "timevae_best.pt"), map_location=device))
    te = run_epoch(model, test_loader, device, beta)
    print(f"Test (best-val checkpoint): total {te[0]:.1f}, recon {te[1]:.1f}, kl {te[2]:.1f}")

    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k, name in zip(axes, (1, 2), ("Reconstruction", "KL")):
        ax.plot([h[k] for h in history["train"]], label="train")
        ax.plot([h[k] for h in history["val"]], label="val")
        ax.set_title(name); ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "loss_curves.png"))


#------------------------------TimeGAN--------------------------------------

def pretrain_embedder(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0

    for x, _ in loader:
        x = x.to(device)

        with torch.set_grad_enabled(training):
            x_hat = model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def pretrain_supervisor(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0

    for x, _ in loader:
        x = x.to(device)

        with torch.no_grad():
            h = model.embedder(x)

        with torch.set_grad_enabled(training):
            h_hat = model.supervisor(h)

            loss = torch.nn.functional.mse_loss(
                h_hat[:, :-1, :],
                h[:, 1:, :]
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def generator_loss(model, x, z, gamma=1.0):
    with torch.no_grad():
        h = model.embedder(x)

    h_fake_raw = model.generator(z)
    h_fake = model.supervisor(h_fake_raw)
    x_fake = model.recovery(h_fake)

    fake_logits = model.discriminator(h_fake)
    fake_raw_logits = model.discriminator(h_fake_raw)

    criterion = torch.nn.BCEWithLogitsLoss()

    loss_adv = criterion(fake_logits, torch.ones_like(fake_logits))
    loss_adv_raw = criterion(fake_raw_logits, torch.ones_like(fake_raw_logits))

    loss_s = torch.nn.functional.mse_loss(
        model.supervisor(h)[:, :-1, :],
        h[:, 1:, :]
    )

    real_mean = x.mean(dim=(0, 1))
    fake_mean = x_fake.mean(dim=(0, 1))

    real_std = x.std(dim=(0, 1), unbiased=False)
    fake_std = x_fake.std(dim=(0, 1), unbiased=False)

    loss_m = (
            torch.mean(torch.abs(real_mean - fake_mean))
            + torch.mean(torch.abs(real_std - fake_std))
    )

    loss_g = (
            loss_adv
            + gamma * loss_adv_raw
            + 100 * torch.sqrt(loss_s + 1e-8)
            + 100 * loss_m
    )

    return loss_g

def discriminator_loss(model, x, z, gamma=1.0):
    with torch.no_grad():
        h = model.embedder(x)
        h_fake_raw = model.generator(z)
        h_fake = model.supervisor(h_fake_raw)

    real_logits = model.discriminator(h)
    fake_logits = model.discriminator(h_fake)
    fake_raw_logits = model.discriminator(h_fake_raw)

    criterion = torch.nn.BCEWithLogitsLoss()

    loss_real = criterion(real_logits, torch.ones_like(real_logits))
    loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
    loss_fake_raw = criterion(fake_raw_logits, torch.zeros_like(fake_raw_logits))

    loss_d = loss_real + loss_fake + gamma * loss_fake_raw

    return loss_d

def joint_train_epoch():
    pass

def timegan():
    num_epochs, lr = 5, 1e-3

    train_set, val_set, test_set, norm = get_datasets("order_sample.csv")
    # DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    # train_set, val_set, test_set, norm = get_datasets(DATA_PATH)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeGAN().to(device)

    embedder_optimizer = Adam(
        list(model.embedder.parameters()) +
        list(model.recovery.parameters()),
        lr=lr,
    )

    for epoch in range(num_epochs):
        train_loss = pretrain_embedder(model, train_loader, device, embedder_optimizer)
        val_loss = pretrain_embedder(model, val_loader, device)
        print(f"Embedder Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    supervisor_optimizer = Adam(
        model.supervisor.parameters(),
        lr=lr
    )

    for epoch in range(num_epochs):
        train_loss = pretrain_supervisor(model, train_loader, device, supervisor_optimizer)
        val_loss = pretrain_supervisor(model, val_loader, device)

        print(f"Supervisor Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    # x, _ = next(iter(val_loader))
    # x = x.to(device)
    #
    # model.eval()
    #
    # with torch.no_grad():
    #     h = model.embedder(x)
    #     h_hat = model.supervisor(h)
    #
    #     persistence_loss = torch.nn.functional.mse_loss(
    #         h[:, :-1, :],
    #         h[:, 1:, :]
    #     )
    #
    #     supervisor_loss = torch.nn.functional.mse_loss(
    #         h_hat[:, :-1, :],
    #         h[:, 1:, :]
    #     )
    #
    # print("Persistence:", persistence_loss.item())
    # print("Supervisor:", supervisor_loss.item())


def main():
    model_type = "timegan"  # Change to "vae" for baseline

    if model_type == "vae":
        vae()
    elif model_type == "timegan":
        timegan()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()