"""
Command:
 - data/generated/synthetic.npy  (n_samples x L x D) unscaled (original feature units)
"""
import os
import argparse
import joblib
import numpy as np
import torch

# Import model classes (same definitions as train_timegan)
from train_timegan import Generator, Supervisor, Recovery, Embedder, set_seed

def load_models(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get('meta', None)
    if meta is None:
        raise RuntimeError('Checkpoint missing meta data.')
    D = meta['D']
    L = meta['L']
    embed_dim = meta['embed_dim']
    hidden_dim = meta['hidden_dim']
    z_dim = meta['z_dim']
    # instantiate
    G = Generator(z_dim, hidden_dim, embed_dim).to(device)
    S = Supervisor(embed_dim, hidden_dim).to(device)
    R = Recovery(embed_dim, hidden_dim, D).to(device)
    G.load_state_dict(ckpt['G'])
    S.load_state_dict(ckpt['S'])
    R.load_state_dict(ckpt['R'])
    G.eval(); S.eval(); R.eval()
    return G, S, R, meta

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    ckpt_path = os.path.join(args.models_dir, 'timegan_latest.pth')
    if not os.path.exists(ckpt_path):
        # fallback to epoch file
        files = [f for f in os.listdir(args.models_dir) if f.startswith('timegan_epoch')]
        files.sort()
        if len(files) == 0:
            raise RuntimeError('No checkpoint found in models_dir')
        ckpt_path = os.path.join(args.models_dir, files[-1])
    G, S, R, meta = load_models(ckpt_path, device)
    L = meta['L']
    z_dim = meta['z_dim']
    D = meta['D']

    # generate
    n = args.n_samples
    batch = args.batch_size
    all_gen = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        z = torch.randn(b, L, z_dim, device=device)
        with torch.no_grad():
            e_hat = G(z)
            h_hat = S(e_hat)
            x_hat = R(h_hat)  # scaled space
        all_gen.append(x_hat.cpu().numpy())
    gen_scaled = np.concatenate(all_gen, axis=0)  # (n, L, D)

    # inverse scale to original units
    scaler = joblib.load(os.path.join(args.data_dir, 'scaler.pkl'))
    n_all, Ls, Dd = gen_scaled.shape
    flat = gen_scaled.reshape(-1, Dd)
    inv = scaler.inverse_transform(flat).reshape(n_all, Ls, Dd)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'synthetic.npy')
    np.save(out_path, inv)
    print('Saved generated synthetic samples to', out_path, 'shape=', inv.shape)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models_dir', default='models')
    p.add_argument('--data_dir', default='data/processed')
    p.add_argument('--out_dir', default='data/generated')
    p.add_argument('--n_samples', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    generate(args)
