from torch.utils.data import DataLoader
from dataset import get_datasets
from modules import TimeVAE, vae_loss
import torch

train_set, val_set, test_set, norm = get_datasets("order_sample.csv")

model = TimeVAE()
x, _ = next(iter(DataLoader(train_set, batch_size=64)))
x_hat, mu, logvar = model(x)
print(x_hat.shape == x.shape)
print(vae_loss(x_hat, x, mu, logvar))
model.eval()                       # BatchNorm uses running stats when sampling
print(model.sample(4).shape)       # (4, 64, 40)

x, _ = next(iter(DataLoader(train_set, batch_size=64)))
print(x.mean().item(), x.std().item(), x.abs().max().item())
print(x[..., :18].abs().max().item(), x[..., 18:38].abs().max().item(), x[..., 38:].abs().max().item())   # prices vs volumes vs spread

xs, _ = next(iter(DataLoader(train_set, batch_size=64, shuffle=True)))
print(xs.mean().item(), xs.std().item(), xs.abs().max().item())

from dataset import load_orderbook, chronological_split
X = load_orderbook("order_sample.csv")
X_train, X_val, X_test = chronological_split(X)

import numpy as np
x, _ = next(iter(DataLoader(train_set, batch_size=64)))   # the unshuffled first batch
w, t, c = np.unravel_index(x.abs().argmax().item(), x.shape)
print(w, t, c)            # window index, timestep, column
print(x[w, t, c].item())
i = w * 8 + t
print(X_train[i])


model = TimeVAE()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x, _ = next(iter(DataLoader(train_set, batch_size=64, shuffle=True)))
for step in range(300):
    x_hat, mu, logvar = model(x)
    loss, recon, kl = vae_loss(x_hat, x, mu, logvar)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(step, recon.item(), kl.item())