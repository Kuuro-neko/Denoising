"""
Denoising VAE for paired images (noisy -> clean) using PyTorch.

Assumes file layout:
 patches/clean/example.jpg
 patches/noisy/noisy_example.jpg

Creates train/test datasets, trains a VAE, saves checkpoints, and can
run a demo on a held-out noisy image.

Usage example:
 python vae_denoise.py \
    --data-root patches \
    --epochs 20 \
    --batch-size 128 \
    --latent-dim 128 \
    --save-path vae_checkpoint.pth \
    --demo-image patches/noisy/noisy_example.jpg

Technical notes:
 - Encoder/Decoder use small conv nets appropriate for 125x125 images.
 - Loss = MSE(recon, clean) + beta * KL. beta=1 by default.
 - Model is designed to run on GPU if available.

"""

import os
import random
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ----------------------------- Dataset -------------------------------------
class PairedImageDataset(Dataset):
    """Loads paired noisy/clean images.
    Expect noisy files to be named noisy_<basename> and clean files to be <basename>,
    or match by basename after stripping a configurable prefix.
    """
    def __init__(self, clean_dir, noisy_dir, file_list=None, img_size=125, noisy_prefix='noisy_'):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.img_size = img_size
        self.noisy_prefix = noisy_prefix

        # gather basenames from clean dir
        if file_list is None:
            clean_paths = sorted(glob(os.path.join(clean_dir, '*')))
            self.basenames = [os.path.basename(p) for p in clean_paths if os.path.isfile(p)]
        else:
            self.basenames = file_list

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),  # scales to [0,1]
        ])

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        base = self.basenames[idx]
        clean_path = os.path.join(self.clean_dir, base)
        noisy_name = self.noisy_prefix + base
        noisy_path = os.path.join(self.noisy_dir, noisy_name)

        # fallback: try to find any file containing base
        if not os.path.isfile(noisy_path):
            candidates = glob(os.path.join(self.noisy_dir, f'*{os.path.splitext(base)[0]}*'))
            noisy_path = candidates[0] if candidates else None

        if not os.path.isfile(clean_path):
            raise FileNotFoundError(f'Clean image not found: {clean_path}')
        if noisy_path is None or not os.path.isfile(noisy_path):
            raise FileNotFoundError(f'Noisy image not found for base {base} in {self.noisy_dir}')

        clean = Image.open(clean_path).convert('RGB')
        noisy = Image.open(noisy_path).convert('RGB')

        clean_t = self.transform(clean)
        noisy_t = self.transform(noisy)

        return noisy_t, clean_t, base

# ----------------------------- VAE Model ----------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, img_size=125):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        # dynamically determine feature dimension based on actual image size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            h = self.conv(dummy)
            self._feat_dim = h.numel()

        self.fc_mu = nn.Linear(self._feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._feat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=128, img_size=125):
        super().__init__()
        self.img_size = img_size

        # must match encoder's computed feature shape
        self.enc = ConvEncoder(out_channels, latent_dim, img_size)
        self._feat_dim = self.enc._feat_dim
        self._feat_channels = 512

        # compute H and W
        self._feat_hw = self._feat_dim // self._feat_channels

        # square??
        self._feat_side = int(self._feat_hw ** 0.5)

        self.fc = nn.Linear(latent_dim, self._feat_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 512, self._feat_side, self._feat_side)
        x = self.deconv(h)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, img_size=125):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, latent_dim, img_size)
        self.decoder = ConvDecoder(in_channels, latent_dim, img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ----------------------------- Utilities ----------------------------------

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # recon_x and x in [0,1]
    mse = F.mse_loss(recon_x, x, reduction='mean')
    # KL divergence between N(mu, var) and N(0,1)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld, mse.item(), kld.item()

# ----------------------------- Training loop ------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    print(f'Using device: {device}')

    clean_dir = os.path.join(args.data_root, 'clean')
    noisy_dir = os.path.join(args.data_root, 'noisy')

    # all basenames from clean dir
    all_basenames = sorted([os.path.basename(p) for p in glob(os.path.join(clean_dir, '*')) if os.path.isfile(p)])
    if len(all_basenames) == 0:
        raise RuntimeError('No images found in clean dir: ' + clean_dir)

    # shuffle and split with optional cap
    original_total = len(all_basenames)
    print(f'Found images: {original_total} in {clean_dir} (matching noisy assumed in {noisy_dir})')
    print(f'Using noisy prefix: "{args.noisy_prefix}", test fraction: {args.test_frac}')

    random.seed(args.seed)
    random.shuffle(all_basenames)

    # apply optional cap
    max_s = int(args.max_samples) if args.max_samples is not None else 0
    if max_s > 0 and original_total > max_s:
        all_basenames = all_basenames[:max_s]
        print(f'Applied max-samples cap: {max_s} (reduced from {original_total})')

    n_total = len(all_basenames)
    n_test = max(1000, int(n_total * args.test_frac))
    if n_test >= n_total:
        n_test = max(1, n_total // 10)
    n_train = n_total - n_test

    train_list = all_basenames[:n_train]
    test_list = all_basenames[n_train:n_train + n_test]

    train_ds = PairedImageDataset(clean_dir, noisy_dir, file_list=train_list, img_size=args.img_size, noisy_prefix=args.noisy_prefix)
    test_ds = PairedImageDataset(clean_dir, noisy_dir, file_list=test_list, img_size=args.img_size, noisy_prefix=args.noisy_prefix)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Dataset: total={n_total}, train={len(train_ds)}, test={len(test_ds)}")

    model = VAE(in_channels=3, latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.load_path and os.path.isfile(args.load_path):
        ck = torch.load(args.load_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        start_epoch = ck.get('epoch', 0) + 1
        print(f'Loaded checkpoint {args.load_path} starting epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_kld = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs-1} [train]')
        for noisy, clean, _ in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(noisy)
            loss, mse_val, kld_val = loss_function(recon, clean, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)
            running_mse += mse_val * noisy.size(0)
            running_kld += kld_val * noisy.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mse': f'{mse_val:.4f}', 'kld': f'{kld_val:.4f}'})

        epoch_loss = running_loss / len(train_ds)
        epoch_mse = running_mse / len(train_ds)
        epoch_kld = running_kld / len(train_ds)
        print(f'[Epoch {epoch}] Train loss: {epoch_loss:.6f}, mse: {epoch_mse:.6f}, kld: {epoch_kld:.6f}')

        # test
        model.eval()
        test_loss = 0.0
        test_mse = 0.0
        test_kld = 0.0
        with torch.no_grad():
            for noisy, clean, _ in tqdm(test_loader, desc=f'Epoch {epoch}/{args.epochs-1} [test]'):
                noisy = noisy.to(device)
                clean = clean.to(device)
                recon, mu, logvar = model(noisy)
                loss, mse_val, kld_val = loss_function(recon, clean, mu, logvar, beta=args.beta)
                test_loss += loss.item() * noisy.size(0)
                test_mse += mse_val * noisy.size(0)
                test_kld += kld_val * noisy.size(0)

        test_loss = test_loss / len(test_ds)
        test_mse = test_mse / len(test_ds)
        test_kld = test_kld / len(test_ds)
        print(f'[Epoch {epoch}] Test  loss: {test_loss:.6f}, mse: {test_mse:.6f}, kld: {test_kld:.6f}')

        # save checkpoint
        ck = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
        save_path = args.save_path or f'vae_epoch_{epoch}.pth'
        torch.save(ck, save_path)
        print(f'Saved checkpoint to {save_path}')

    # final model save
    final_path = args.save_path or 'vae_final.pth'
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': args.epochs-1}, final_path)
    print('Training complete. Final model saved to', final_path)

    # demo inference if requested
    if args.demo_image:
        demo_out = run_demo(model, args.demo_image, device, args.img_size, args.noisy_prefix)
        out_path = args.demo_out or 'demo_out.png'
        demo_out.save(out_path)
        print('Demo output saved to', out_path)

    return model

# ----------------------------- Demo ---------------------------------------

def run_demo(model, noisy_image_path, device, img_size=125, noisy_prefix='noisy_'):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    im = Image.open(noisy_image_path).convert('RGB')
    t = transform(im).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _ = model(t)
    recon = recon.clamp(0,1).cpu().squeeze(0)
    # to PIL
    to_pil = transforms.ToPILImage()
    out = to_pil(recon)
    return out

# ----------------------------- Main / CLI ---------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='patches', help='root directory containing clean/ and noisy/')
    p.add_argument('--img-size', type=int, default=64)
    p.add_argument('--noisy-prefix', type=str, default='noisy_', help='prefix used for noisy files (default noisy_)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--latent-dim', type=int, default=128)
    p.add_argument('--beta', type=float, default=1.0, help='weight for KL term')
    p.add_argument('--save-path', type=str, default='vae_checkpoint.pth')
    p.add_argument('--load-path', type=str, default='')
    p.add_argument('--device', type=str, choices=['cpu','cuda'], default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--test-frac', type=float, default=0.02)
    p.add_argument('--max-samples', type=int, default=2000,
                   help='Maximum number of samples to use from the dataset. Set to 0 for no limit.')
    p.add_argument('--demo-image', type=str, default='')
    p.add_argument('--demo-out', type=str, default='demo_out.png')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # quick sanity check
    assert os.path.isdir(os.path.join(args.data_root, 'clean')), 'patches/clean must exist'
    assert os.path.isdir(os.path.join(args.data_root, 'noisy')), 'patches/noisy must exist'
    train(args)
