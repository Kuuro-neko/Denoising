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
#import torch_directml

# ============= Configuration du device (AMD/CPU) =============
def get_device():
    """Détecte le meilleur device disponible"""
    if torch.cuda.is_available():
        print(f"Device détecté: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Device détecté: CPU")
        return torch.device("cpu")

device = get_device()
#device = torch_directml.device()


# ============= Dataset personnalisé =============
class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noise_types=['gauss', 'poisson', 'sap', 'speckle'], 
                 transform=None, num_images=512):
        """
        Args:
            clean_dir: Répertoire contenant les images propres et bruitées
            noise_types: Liste des préfixes de bruit
            transform: Transformations à appliquer
            num_images: Nombre d'images propres (1 à num_images)
        """
        self.clean_dir = Path(clean_dir)
        self.noise_types = noise_types
        self.transform = transform
        self.num_images = num_images
        
        # Créer la liste de toutes les paires (image bruitée, image propre)
        # ATTENTION : les versions bruitées des patchs sont cherchées dans le même dossier et reconnues uniquement avec leur préfixe
        # Il faudra modifier le script de découpage/bruitage pour qu'il les génère de la même manière
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
        
        print(f"Dataset chargé: {len(self.samples)} paires d'images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        noisy_path, clean_path = self.samples[idx]
        
        # Charger les images
        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img


# ============= Architecture U-Net =============
class DoubleConv(nn.Module):
    """Bloc de deux convolutions avec BatchNorm et ReLU"""
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
    """Architecture U-Net pour le débruitage"""
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Partie descendante (encodeur)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Partie ascendante (décodeur)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Goulot d'étranglement
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Couche finale
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Descente
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Montée
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            # Gérer les dimensions impaires
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            concat = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat)
        
        return self.final_conv(x)


# ============= Entraînement =============
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, save_path='unet_denoiser.pth'):
    """Entraîne le modèle U-Net"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        
        print("Entraînement en cours...")
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Afficher la progression
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0 or (batch_idx + 1) == len(train_loader):
                progress = (batch_idx + 1) / len(train_loader) * 100
                current_loss = train_loss / (batch_idx + 1)
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] ({progress:.1f}%) - Loss: {current_loss:.6f}")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        
        print("\nValidation en cours...")
        with torch.no_grad():
            for batch_idx, (noisy, clean) in enumerate(val_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                val_loss += loss.item()
                
                # Afficher la progression
                if (batch_idx + 1) % max(1, len(val_loader) // 5) == 0 or (batch_idx + 1) == len(val_loader):
                    progress = (batch_idx + 1) / len(val_loader) * 100
                    current_loss = val_loss / (batch_idx + 1)
                    print(f"  Batch [{batch_idx+1}/{len(val_loader)}] ({progress:.1f}%) - Loss: {current_loss:.6f}")
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Ajuster le taux d'apprentissage
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'─'*60}")
        print(f"Résumé Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Modèle sauvegardé (meilleure validation loss: {best_val_loss:.6f})")
        print(f"{'─'*60}")
    
    return train_losses, val_losses


# ============= Visualisation =============
def visualize_results(model, dataset, num_samples=3):
    """Affiche des exemples de débruitage"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            noisy, clean = dataset[idx]
            noisy_input = noisy.unsqueeze(0).to(device)
            denoised = model(noisy_input).cpu().squeeze(0)
            
            # Convertir en images affichables
            noisy_img = noisy.permute(1, 2, 0).numpy()
            clean_img = clean.permute(1, 2, 0).numpy()
            denoised_img = denoised.permute(1, 2, 0).numpy()
            
            # Clipper les valeurs
            noisy_img = np.clip(noisy_img, 0, 1)
            clean_img = np.clip(clean_img, 0, 1)
            denoised_img = np.clip(denoised_img, 0, 1)
            
            axes[i, 0].imshow(noisy_img)
            axes[i, 0].set_title('Image bruitée')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(denoised_img)
            axes[i, 1].set_title('Débruitée')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(clean_img)
            axes[i, 2].set_title('Image propre')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============= Script principal =============
def main():
    # Configuration
    DATA_DIR = "./patches"  # À adapter selon le dossier contenant vos patchs
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    TRAIN_SPLIT = 0.8
    RESUME_TRAINING = False  # Mettre True pour continuer un entraînement
    MODEL_PATH = "unet_denoiser.pth"  # Chemin du modèle à charger
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Charger le dataset
    full_dataset = DenoisingDataset(
        clean_dir=DATA_DIR,
        noise_types=['gauss', 'poisson', 'sap', 'speckle'],
        transform=transform,
        num_images=2533 # À adapter au nombre de patchs dans le dossier /patches
    )
    
    # Diviser en train/val
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Créer le modèle
    model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Charger les poids si on continue un entraînement
    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        print(f"\n=== Chargement du modèle existant: {MODEL_PATH} ===")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✓ Modèle chargé avec succès! Continuation de l'entraînement...\n")
    else:
        print(f"\nModèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Entraîner
    print("\n=== Début de l'entraînement ===\n")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    
    # Visualiser les résultats
    print("\n=== Visualisation des résultats ===\n")
    model.load_state_dict(torch.load('unet_denoiser.pth'))
    visualize_results(model, full_dataset, num_samples=3)
    
    # Plot des pertes
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Évolution de la perte durant l\'entraînement')
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()