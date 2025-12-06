"""
Command:
 python train_timegan.py --data_dir data/processed --models_dir models --epochs 2 --batch_size 2
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# Model blocks (GRU-based)
# -------------------------
class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, embed_dim)
    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)  # (B,L,embed_dim)

class Recovery(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, h):
        h_out, _ = self.gru(h)
        return self.fc(h_out)

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, embed_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(z_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embed_dim)
    def forward(self, z):
        h, _ = self.gru(z)
        return self.fc(h)

class Supervisor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embed_dim)
    def forward(self, h):
        h_out, _ = self.gru(h)
        return self.fc(h_out)

class Discriminator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        h_out, _ = self.gru(h)
        # Pool over time
        pooled = torch.mean(h_out, dim=1)
        return torch.sigmoid(self.fc(pooled)).view(-1)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_loader(seqs, batch_size=128):
    tensor = torch.tensor(seqs, dtype=torch.float32)
    ds = TensorDataset(tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

# -------------------------
# Training loop
# -------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # load data & meta
    seqs = np.load(os.path.join(args.data_dir, 'sequences.npy'))
    with open(os.path.join(args.data_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)
    N, L, D = seqs.shape
    print('Loaded sequences:', seqs.shape, 'columns:', meta.get('columns', []))

    # model dims
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    z_dim = args.z_dim

    # instantiate models
    E = Embedder(D, hidden_dim, embed_dim).to(device)
    R = Recovery(embed_dim, hidden_dim, D).to(device)
    G = Generator(z_dim, hidden_dim, embed_dim).to(device)
    S = Supervisor(embed_dim, hidden_dim).to(device)
    Dscr = Discriminator(embed_dim, hidden_dim).to(device)

    # optimizers
    opt_E = optim.Adam(list(E.parameters()) + list(R.parameters()), lr=args.lr)
    opt_GS = optim.Adam(list(G.parameters()) + list(S.parameters()), lr=args.lr)
    opt_D = optim.Adam(Dscr.parameters(), lr=args.lr)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    loader = build_loader(seqs, args.batch_size)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = {'recon':0.0, 'sup':0.0, 'd':0.0, 'g':0.0}
        for (batch,) in loader:
            batch = batch.to(device)  # (B,L,D)
            B = batch.size(0)

            # -----------------------
            # 1) Reconstruction phase (E + R)
            # -----------------------
            E.zero_grad(); R.zero_grad()
            h = E(batch)  # (B,L,embed)
            X_tilde = R(h)
            loss_recon = mse(X_tilde, batch)
            loss_recon.backward()
            opt_E.step()
            epoch_losses['recon'] += loss_recon.item()

            # -----------------------
            # 2) Supervised training (S)
            # -----------------------
            E.eval()
            with torch.no_grad():
                h_full = E(batch)
            E.train()
            # h_full: (B,L,embed), we supervise h[:, :-1] -> h[:, 1:]
            h_input = h_full[:, :-1, :].detach()
            h_target = h_full[:, 1:, :].detach()
            S.zero_grad(); G.zero_grad()
            h_pred = S(h_input)
            loss_sup = mse(h_pred, h_target)
            loss_sup.backward()
            opt_GS.step()
            epoch_losses['sup'] += loss_sup.item()

            # -----------------------
            # 3) Adversarial training
            # -----------------------
            # prepare real embeddings
            with torch.no_grad():
                h_real = E(batch).detach()  # (B,L,embed)

            # 3a) Discriminator update
            # create synthetic embedding via G(z) -> S(...)
            z = torch.randn(B, L, z_dim, device=device)
            e_hat = G(z)
            h_hat = S(e_hat)  # supervised generator output
            Dscr.zero_grad()
            y_real = Dscr(h_real)
            y_fake = Dscr(h_hat.detach())
            # labels
            real_labels = torch.ones_like(y_real, device=device)
            fake_labels = torch.zeros_like(y_fake, device=device)
            loss_d = bce(y_real, real_labels) + bce(y_fake, fake_labels)
            loss_d.backward()
            opt_D.step()
            epoch_losses['d'] += loss_d.item()

            # 3b) Generator (G+S) update
            G.zero_grad(); S.zero_grad()
            y_fake_for_g = Dscr(h_hat)
            loss_g_adv = bce(y_fake_for_g, real_labels)  # fool D
            # supervised loss again (encourage temporal dynamics)
            loss_g_sup = mse(h_hat[:, :-1, :], h_hat[:, 1:, :]) if h_hat.size(1) > 1 else torch.tensor(0.0, device=device)
            # moment loss: match means and variances (simple)
            mean_real = torch.mean(h_real, dim=[0,1])
            mean_fake = torch.mean(h_hat, dim=[0,1])
            var_real = torch.var(h_real, dim=[0,1])
            var_fake = torch.var(h_hat, dim=[0,1])
            loss_mom = mse(mean_fake, mean_real) + mse(var_fake, var_real)
            loss_g = loss_g_adv + args.gamma * loss_mom + args.alpha * loss_g_sup
            loss_g.backward()
            opt_GS.step()
            epoch_losses['g'] += loss_g.item()

        # average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= len(loader)
        if epoch % args.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs} | recon: {epoch_losses['recon']:.6f} sup: {epoch_losses['sup']:.6f} d: {epoch_losses['d']:.6f} g: {epoch_losses['g']:.6f}")

        # Save checkpoint occasionally
        if epoch % args.save_every == 0 or epoch == args.epochs:
            os.makedirs(args.models_dir, exist_ok=True)
            ckpt = {
                'E': E.state_dict(),
                'R': R.state_dict(),
                'G': G.state_dict(),
                'S': S.state_dict(),
                'D': Dscr.state_dict(),
                'meta': {'D': D, 'L': L, 'embed_dim': embed_dim, 'hidden_dim': hidden_dim, 'z_dim': z_dim}
            }
            torch.save(ckpt, os.path.join(args.models_dir, 'timegan_epoch%03d.pth' % epoch))
            torch.save(ckpt, os.path.join(args.models_dir, 'timegan_latest.pth'))
            print('Saved checkpoint at epoch', epoch)

    print('Training finished. Final checkpoint in', args.models_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/processed')
    p.add_argument('--models_dir', default='models')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--z_dim', type=int, default=32)
    p.add_argument('--gamma', type=float, default=1.0, help='moment loss weight')
    p.add_argument('--alpha', type=float, default=1.0, help='supervised loss weight in generator')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--print_every', type=int, default=5)
    p.add_argument('--save_every', type=int, default=50)
    args = p.parse_args()
    train(args)
