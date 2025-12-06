#!/usr/bin/env python3
"""
evaluate.py (robust)
Performs evaluations between real sequences and generated samples:
 - MMD (RBF)
 - Optional DTW (fastdtw)
 - ACF/PACF comparison
 - Predictive utility: train LSTM on synthetic -> test on real
Usage:
 python evaluate.py --real data/processed/sequences.npy --synth data/generated/synthetic.npy --out_dir results
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# optional imports
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    FASTDTW = True
except Exception:
    FASTDTW = False

# statsmodels ACF/PACF
try:
    from statsmodels.tsa.stattools import acf, pacf
except Exception:
    acf = None
    pacf = None

# -------------------------
# MMD (RBF)
# -------------------------
def compute_mmd(X, Y, sigma=1.0):
    # X, Y: (n_samples, flattened_dim)
    if X.size == 0 or Y.size == 0:
        raise ValueError("Empty arrays passed to compute_mmd")
    XX = cdist(X, X, 'sqeuclidean')
    YY = cdist(Y, Y, 'sqeuclidean')
    XY = cdist(X, Y, 'sqeuclidean')
    Kxx = np.exp(-XX/(2*sigma**2)).mean()
    Kyy = np.exp(-YY/(2*sigma**2)).mean()
    Kxy = np.exp(-XY/(2*sigma**2)).mean()
    return float(Kxx + Kyy - 2*Kxy)

# -------------------------
# DTW wrapper (fixed)
# -------------------------
def mean_dtw(real, synth, feature_idx=3, n_pairs=200):
    if not FASTDTW:
        print('[DTW] fastdtw not installed; skipping DTW. Install fastdtw for DTW (pip install fastdtw).')
        return None

    n = min(len(real), len(synth))
    if n == 0:
        print('[DTW] No sequences to compare.')
        return None

    n_pairs = min(n_pairs, n)
    idxs = random.sample(range(n), n_pairs)
    dists = []

    for i in idxs:
        r = real[i]
        s = synth[i]

        # safety checks
        if r.ndim != 2 or s.ndim != 2:
            continue
        if feature_idx >= r.shape[1] or feature_idx >= s.shape[1]:
            continue

        # extract 1-D feature sequences
        r_feat = r[:, feature_idx]
        s_feat = s[:, feature_idx]

        # FIX: do not index again -> pass directly
        d, _ = fastdtw(r_feat, s_feat, dist=euclidean)
        dists.append(d)

    if len(dists) == 0:
        return None

    return float(np.mean(dists)), float(np.std(dists))


# -------------------------
# Predictive utility (small LSTM)
# -------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h[:, -1, :])
        return out

def train_forecaster(train_seqs, valid_seqs, feature_idx, epochs=30, batch_size=64, lr=1e-3, device='cpu'):
    # checks
    if train_seqs.shape[0] == 0:
        raise ValueError("No training sequences provided.")
    X = train_seqs[:, :-1, :]
    y = train_seqs[:, -1, feature_idx]
    Xv = valid_seqs[:, :-1, :]
    yv = valid_seqs[:, -1, feature_idx]

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32))
    vdl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimpleLSTM(X.shape[2], hidden_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in vdl:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(val_losses)) if len(val_losses)>0 else float('nan')
        if val_loss < best_val:
            best_val = val_loss
        if ep % 10 == 0 or ep == epochs:
            print(f'[Forecaster] epoch {ep} val_mse={val_loss:.6f}')
    return model, best_val

def evaluate_predictive(synthetic, real, test_fraction=0.2, feature_idx=3, device='cpu'):
    n_synth = synthetic.shape[0]
    n_real = real.shape[0]
    if n_synth < 2 or n_real < 2:
        raise ValueError("Need at least 2 sequences in both synth and real for predictive evaluation.")
    idxs_s = np.arange(n_synth); np.random.shuffle(idxs_s)
    idxs_r = np.arange(n_real); np.random.shuffle(idxs_r)
    cut_s = int((1-test_fraction) * n_synth)
    cut_r = int((1-test_fraction) * n_real)
    if cut_s == 0 or cut_r == n_real:
        # avoid empty train/val/test splits; reduce test_fraction
        cut_s = max(1, cut_s)
    train_s = synthetic[idxs_s[:cut_s]]
    val_s = synthetic[idxs_s[cut_s:]] if cut_s < n_synth else synthetic[idxs_s[:1]]
    test_r = real[idxs_r[cut_r:]] if cut_r < n_real else real[idxs_r[-1:]]
    model, val_mse = train_forecaster(train_s, val_s, feature_idx, epochs=30, device=device)
    model.eval()
    X_test = torch.tensor(test_r[:, :-1, :], dtype=torch.float32).to(device)
    y_test = torch.tensor(test_r[:, -1, feature_idx], dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        pred = model(X_test)
        mse = nn.MSELoss()(pred, y_test).item()
    return float(val_mse), float(mse)

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real', default='data/processed/sequences.npy')
    p.add_argument('--synth', default='data/generated/synthetic.npy')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--feature_index', type=int, default=3, help='index of Close in columns (0-based)')
    p.add_argument('--mmd_sigma', type=float, default=1.0)
    p.add_argument('--n_dtw_pairs', type=int, default=200)
    p.add_argument('--n_sub', type=int, default=200, help='subsample for MMD to limit memory')
    p.add_argument('--device', type=str, default=None, help='cpu or cuda; autodetect if not set')
    args = p.parse_args()

    # device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    os.makedirs(args.out_dir, exist_ok=True)

    # load arrays with validation
    if not os.path.exists(args.real):
        raise FileNotFoundError(f"Real file not found: {args.real}")
    if not os.path.exists(args.synth):
        raise FileNotFoundError(f"Synth file not found: {args.synth}")

    real = np.load(args.real)
    synth = np.load(args.synth)
    print('[INFO] Real shape', real.shape, 'Synth shape', synth.shape)

    # basic shape checks
    if real.ndim != 3 or synth.ndim != 3:
        raise ValueError("Expected real and synth arrays to be 3D (N, L, D). Got shapes: real=%s synth=%s"%(real.shape, synth.shape))
    N, L, D = real.shape
    if args.feature_index >= D:
        raise IndexError(f"feature_index {args.feature_index} >= num features {D}")

    # MMD
    X = real.reshape(real.shape[0], -1)
    Y = synth.reshape(synth.shape[0], -1)
    # subsample
    n_sub = min(args.n_sub, X.shape[0])
    idxs = np.random.choice(X.shape[0], n_sub, replace=False) if X.shape[0] > n_sub else np.arange(X.shape[0])
    Xs = X[idxs]
    n_sub_y = min(args.n_sub, Y.shape[0])
    idxs_y = np.random.choice(Y.shape[0], n_sub_y, replace=False) if Y.shape[0] > n_sub_y else np.arange(Y.shape[0])
    Ys = Y[idxs_y]
    try:
        mmd_val = compute_mmd(Xs, Ys, sigma=args.mmd_sigma)
    except Exception as e:
        print('[MMD] computation failed:', e)
        mmd_val = None
    print('[MMD] value =', mmd_val)

    # DTW
    dtw_stats = mean_dtw(real, synth, feature_idx=args.feature_index, n_pairs=args.n_dtw_pairs)
    print('[DTW] stats =', dtw_stats)

    # ACF / PACF (safe fallback)
    lags = 40
    acf_real = acf_synth = pacf_real = pacf_synth = None
    try:
        if acf is not None:
            real_series = real.reshape(-1, real.shape[2])[:, args.feature_index]
            synth_series = synth.reshape(-1, synth.shape[2])[:, args.feature_index]
            acf_real = acf(real_series, nlags=lags, fft=True)
            acf_synth = acf(synth_series, nlags=lags, fft=True)
    except Exception as e:
        print('[ACF] failed:', e)
    try:
        if pacf is not None:
            # try default, then fallback to method='ywm' if problem
            pacf_real = pacf(real_series, nlags=lags)
            pacf_synth = pacf(synth_series, nlags=lags)
    except Exception as e:
        try:
            pacf_real = pacf(real_series, nlags=lags, method='ywm')
            pacf_synth = pacf(synth_series, nlags=lags, method='ywm')
        except Exception as e2:
            print('[PACF] failed:', e, e2)

    # safe plotting (only plot available data)
    try:
        if acf_real is not None and acf_synth is not None:
            plt.figure(figsize=(10,4))
            plt.plot(acf_real, label='real acf')
            plt.plot(acf_synth, label='synth acf')
            plt.legend(); plt.title('ACF comparison')
            plt.savefig(os.path.join(args.out_dir,'acf_compare.png')); plt.close()
        if pacf_real is not None and pacf_synth is not None:
            plt.figure(figsize=(10,4))
            plt.plot(pacf_real, label='real pacf')
            plt.plot(pacf_synth, label='synth pacf')
            plt.legend(); plt.title('PACF comparison')
            plt.savefig(os.path.join(args.out_dir,'pacf_compare.png')); plt.close()
    except Exception as e:
        print('[PLOT] saving ACF/PACF failed:', e)

    # Predictive utility
    try:
        val_mse, test_mse = evaluate_predictive(synth, real, feature_idx=args.feature_index, device=device)
        print('[Forecaster] val_mse(on synth) =', val_mse, 'test_mse(on real) =', test_mse)
    except Exception as e:
        print('[Forecaster] predictive evaluation failed:', e)
        val_mse, test_mse = None, None

    # summary
    res = {
        'mmd': mmd_val,
        'dtw_mean': dtw_stats[0] if dtw_stats is not None else None,
        'dtw_std': dtw_stats[1] if dtw_stats is not None else None,
        'forecaster_val_mse': val_mse,
        'forecaster_test_mse': test_mse,
    }
    with open(os.path.join(args.out_dir, 'eval_summary.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('[INFO] Saved evaluation summary to', os.path.join(args.out_dir,'eval_summary.json'))

if __name__ == '__main__':
    main()
