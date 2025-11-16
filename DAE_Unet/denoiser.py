import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ============= Importer l'architecture U-Net =============
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
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
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
            
            concat = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat)
        
        return self.final_conv(x)


# ============= SCRIPT 1 : Débruiter une image =============
def denoise_image(image_path, model_path="unet_denoiser.pth", save_path=None):
    """
    Applique le modèle de débruitage sur une image
    
    Args:
        image_path: Chemin vers l'image à débruiter
        model_path: Chemin vers le fichier .pth du modèle
        save_path: Chemin pour sauvegarder l'image débruitée (optionnel)
    """
    # Device
    try:
        import torch_directml
        device = torch_directml.device()
        print("Utilisation de DirectML")
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modèle chargé")
    
    # Charger et préparer l'image
    print(f"Chargement de l'image {image_path}...")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Débruitage
    print("Débruitage en cours...")
    with torch.no_grad():
        denoised_tensor = model(image_tensor)
    
    # Convertir en image
    denoised_tensor = denoised_tensor.cpu().squeeze(0)
    denoised_array = denoised_tensor.permute(1, 2, 0).numpy()
    denoised_array = np.clip(denoised_array, 0, 1)
    denoised_image = Image.fromarray((denoised_array * 255).astype(np.uint8))
    
    # Sauvegarder si demandé
    if save_path:
        denoised_image.save(save_path)
        print(f"Image débruitée sauvegardée dans {save_path}")
    
    # Afficher la comparaison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Image bruitée (originale)')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_image)
    axes[1].set_title('Image débruitée')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Terminé !")
    return denoised_image


# ============= SCRIPT 3 : Débruiter un dossier entier =============
def denoise_folder(input_folder, output_folder, model_path="unet_denoiser.pth"):
    """
    Débruite toutes les images d'un dossier
    
    Args:
        input_folder: Dossier contenant les images bruitées
        output_folder: Dossier où sauvegarder les images débruitées
        model_path: Chemin vers le fichier .pth du modèle
    """
    import os
    from pathlib import Path
    
    # Créer le dossier de sortie s'il n'existe pas
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Device
    try:
        import torch_directml
        device = torch_directml.device()
        print("Utilisation de DirectML")
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modèle chargé!\n")
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Lister les images
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Traitement de {len(image_files)} images...\n")
    
    # Traiter chaque image
    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"denoised_{filename}")
        
        try:
            # Charger l'image
            image = Image.open(input_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Débruiter
            with torch.no_grad():
                denoised_tensor = model(image_tensor)
            
            # Convertir et sauvegarder
            denoised_tensor = denoised_tensor.cpu().squeeze(0)
            denoised_array = denoised_tensor.permute(1, 2, 0).numpy()
            denoised_array = np.clip(denoised_array, 0, 1)
            denoised_image = Image.fromarray((denoised_array * 255).astype(np.uint8))
            denoised_image.save(output_path)
            
            print(f"[{idx}/{len(image_files)}] {filename} -> {output_path}")
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] Erreur avec {filename}: {e}")
    
    print(f"\nTraitement terminé ! Images sauvegardées dans {output_folder}")


# ============= EXEMPLES D'UTILISATION =============
if __name__ == "__main__":
    print("=" * 60)
    print("SCRIPTS DE DÉBRUITAGE - Choisissez une option:")
    print("=" * 60)
    print("1. Débruiter une seule image")
    print("2. Débruiter un dossier entier")
    print("=" * 60)
    
    choice = input("\nVotre choix (1/2): ")
    
    if choice == "1":
        print("\n--- Débruitage d'une image ---")
        image_path = input("Chemin de l'image à débruiter: ")
        save_path = input("Chemin de sauvegarde (appuyez sur Entrée pour ne pas sauvegarder): ")
        save_path = save_path if save_path else None
        denoise_image(image_path, save_path=save_path)
        
    elif choice == "2":
        print("\n--- Débruitage d'un dossier ---")
        input_folder = input("Dossier contenant les images bruitées: ")
        output_folder = input("Dossier de sortie pour les images débruitées: ")
        denoise_folder(input_folder, output_folder)
        
    else:
        print("Choix invalide!")
