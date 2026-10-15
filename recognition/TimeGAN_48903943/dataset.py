import numpy as np
import joblib
import torch
from sklearn.preprocessing import StandardScaler

CENT = 100  # LOBSTER prices are dollars * 10000, so 1 cent = 100 units


def load_orderbook(path):
    # LOBSTER orderbook files have no header: 40 columns, one row per event
    return np.loadtxt(path, delimiter=",", dtype=np.float64)

def chronological_split(X, train=0.7, val=0.15):
    n = len(X)
    i, j = int(n * train), int(n * (train + val))
    return X[:i], X[i:j], X[j:]

def make_features(X):
    X = np.asarray(X, dtype=np.float64)  # keep float64 until after subtracting the mid
    ask_p, ask_v = X[:, 0::4], X[:, 1::4]
    bid_p, bid_v = X[:, 2::4], X[:, 3::4]

    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2   # average of the best ask price and best bid price
    prices = np.concatenate(
        [ask_p - mid[:, None], bid_p - mid[:, None]], axis=1
    ) / CENT                                    # offsets from mid, in cents
    volumes = np.log1p(np.concatenate([ask_v, bid_v], axis=1))
    return prices, volumes, mid


class LOBNormaliser:
    def fit(self, X_train):
        P, V, _ = make_features(X_train)
        self.p_scaler = StandardScaler().fit(P)
        self.v_scaler = StandardScaler().fit(V)
        return self

    def transform(self, X):
        P, V, mid = make_features(X)
        Z = np.concatenate(
            [self.p_scaler.transform(P), self.v_scaler.transform(V)], axis=1
        ).astype(np.float32)
        return Z, mid                           # mid is needed for the inverse

    def inverse(self, Z, mid):
        Z = np.asarray(Z, dtype=np.float64)
        P = self.p_scaler.inverse_transform(Z[:, :20]) * CENT + mid[:, None]
        V = np.clip(np.expm1(self.v_scaler.inverse_transform(Z[:, 20:])), 0, None)
        X = np.empty((len(Z), 40))
        X[:, 0::4], X[:, 1::4] = P[:, :10], V[:, :10]   # ask price, ask vol
        X[:, 2::4], X[:, 3::4] = P[:, 10:], V[:, 10:]   # bid price, bid vol
        return X

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

# Testing for transformation error
# norm = LOBNormaliser().fit(X_train)
# Z, mid = norm.transform(X_val)
# err = np.abs(norm.inverse(Z, mid) - X_val)
# print(err[:, 0::2].max(), err[:, 1::2].max())  # price cols, volume cols

class LOBWindows(torch.utils.data.Dataset):
    def __init__(self, Z, mid, window=64, stride=8):
        self.Z = torch.from_numpy(Z)              # (n_events, 40), already normalised
        self.mid = torch.from_numpy(mid)          # (n_events,)
        self.window, self.stride = window, stride
        self.n = (len(Z) - window) // stride + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        return self.Z[s:s + self.window], self.mid[s:s + self.window]


def get_datasets(path, window=64, stride=8):
    X = load_orderbook(path)
    X_train, X_val, X_test = chronological_split(X)
    norm = LOBNormaliser().fit(X_train)            # train split only
    sets = []
    for part in (X_train, X_val, X_test):
        Z, mid = norm.transform(part)
        sets.append(LOBWindows(Z, mid, window, stride))
    return (*sets, norm)

# main
if __name__ == "__main__":
    train_set, val_set, test_set, norm = get_datasets("order_sample.csv")
    print(len(train_set), len(val_set), len(test_set))
    x, m = train_set[0]
    print(x.shape, m.shape)          # (64, 40), (64,)