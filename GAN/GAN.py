import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ======================================
# DEVICE SETUP
# ======================================
def get_device():
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = get_device()


# ======================================
# DATASET
# ======================================
class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noise_types=['gauss', 'poisson', 'sap', 'speckle'],
                 transform=None, num_images=512):
        self.clean_dir = Path(clean_dir)
        self.noise_types = noise_types
        self.transform = transform
        self.num_images = num_images

        self.samples = []
        for i in range(1, num_images + 1):
            clean_name = f"{i:06d}.jpg"
            clean_path = self.clean_dir / clean_name

            if clean_path.exists():
                for noise in noise_types:
                    noisy_name = f"{noise}_{i:06d}.jpg"
                    noisy_path = self.clean_dir / noisy_name

                    if noisy_path.exists():
                        self.samples.append((noisy_path, clean_path))

        print(f"Dataset loaded: {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.samples[idx]

        noisy = Image.open(noisy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)

        return noisy, clean

# ====================================
# PSNR FUNCTION
# ====================================
def compute_psnr(img1, img2):
    """
    img1, img2: torch tensors [C,H,W] in range [0,1]
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr


# ======================================
# U-NET GENERATOR
# ======================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)


# ======================================
# DISCRIMINATOR (PATCHGAN)
# ======================================
class Discriminator(nn.Module):
    """
    PatchGAN discriminator for conditional GAN (Pix2Pix)
    Input: [noisy_image, clean_or_generated]
    Output: patch-based real/fake map
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        layers = []
        curr_in = in_channels * 2  # concatenated inputs

        for i, feature in enumerate(features):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_in, feature, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(feature) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            curr_in = feature

        layers.append(nn.Conv2d(curr_in, 1, kernel_size=4, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, noisy, clean_or_fake):
        x = torch.cat([noisy, clean_or_fake], dim=1)
        return self.model(x)


# ======================================
# TRAINING LOOP (GAN)
# ======================================
def train_model(
    generator, discriminator, train_loader, val_loader,
    epochs=20, lr=2e-4, lambda_mse=100,
    save_path="unet_gan_generator.pth"
):
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_MSE = nn.MSELoss()

    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        print(f"\n========== EPOCH {epoch+1}/{epochs} ==========")

        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)

            # -----------------------------------------------------
            # Train Discriminator
            # -----------------------------------------------------
            fake = generator(noisy).detach()

            pred_real = discriminator(noisy, clean)
            pred_fake = discriminator(noisy, fake)

            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)

            loss_D_real = criterion_GAN(pred_real, real_labels)
            loss_D_fake = criterion_GAN(pred_fake, fake_labels)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # -----------------------------------------------------
            # Train Generator
            # -----------------------------------------------------
            fake = generator(noisy)
            pred_fake_for_G = discriminator(noisy, fake)

            loss_G_GAN = criterion_GAN(pred_fake_for_G, real_labels)
            loss_G_MSE = criterion_MSE(fake, clean)

            loss_G = loss_G_GAN + lambda_mse * loss_G_MSE

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if (batch_idx + 1) % (len(train_loader) // 5 + 1) == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"D={loss_D.item():.4f} | "
                      f"G={loss_G.item():.4f} "
                      f"(GAN={loss_G_GAN.item():.4f}, MSE={loss_G_MSE.item():.4f})")

        torch.save(generator.state_dict(), save_path)
        print(f"✓ Saved generator to: {save_path}")


# ======================================
# VISUALIZATION
# ======================================
def visualize_results(generator, dataset, num_samples=5):
    generator.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            noisy, clean = dataset[idx]
            inp = noisy.unsqueeze(0).to(device)
            denoised = generator(inp).cpu().squeeze(0)

            # Compute PSNR
            psnr_noisy = compute_psnr(noisy, clean)
            psnr_denoised = compute_psnr(denoised, clean)

            # Convert to displayable images
            noisy_img = noisy.permute(1, 2, 0).numpy()
            clean_img = clean.permute(1, 2, 0).numpy()
            denoised_img = denoised.permute(1, 2, 0).numpy()

            noisy_img = np.clip(noisy_img, 0, 1)
            clean_img = np.clip(clean_img, 0, 1)
            denoised_img = np.clip(denoised_img, 0, 1)

            # ------------------
            # Display images
            # ------------------
            axes[i, 0].imshow(noisy_img)
            axes[i, 0].set_title(f"Noisy\nPSNR={psnr_noisy:.2f} dB")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(denoised_img)
            axes[i, 1].set_title(f"Denoised\nPSNR={psnr_denoised:.2f} dB")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(clean_img)
            axes[i, 2].set_title("Clean (GT)")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("gan_denoising_results.png", dpi=150)
    plt.show()



# ======================================
# MAIN SCRIPT
# ======================================
def main():
    DATA_DIR = "./image_database/patches"
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 2e-4
    TRAIN_SPLIT = 0.8

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_dataset = DenoisingDataset(
        clean_dir=DATA_DIR,
        noise_types=['gauss', 'poisson', 'sap', 'speckle'],
        transform=transform,
        num_images=2533
    )

    train_len = int(len(full_dataset) * TRAIN_SPLIT)
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    generator = UNet().to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters())}")

    print("\n=== TRAINING GAN ===")
    train_model(generator, discriminator, train_loader, val_loader,
                epochs=EPOCHS, lr=LR)

    print("\n=== VISUALIZATION ===")
    generator.load_state_dict(torch.load("unet_gan_generator.pth"))
    visualize_results(generator, full_dataset, num_samples=3)


if __name__ == "__main__":
    main()
