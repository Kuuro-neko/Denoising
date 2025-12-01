import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pandas as pd
import os

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
    print("3. Débruiter un dossier et afficher des graphiques")
    print("=" * 60)
    
    choice = input("\nVotre choix (1/2/3): ")
    
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

    elif choice == "3":
        print("\n --- Graphiques ---")
        input_folder = "./image_database/sample/"
        output_folder = "./image_database/denoised_sample/"
        denoise_folder(input_folder, output_folder)
        clean_folder = input_folder + "clean/"

        # Initialisation des listes pour les résultats
        noise_types = ['gauss', 'poisson', 'sap', 'speckle']
        psnr_values = {noise: [] for noise in noise_types}
        ssim_values = {noise: [] for noise in noise_types}

        # Parcourir les types de bruit
        for noise in noise_types:
            for filename in os.listdir(clean_folder):
                if filename.endswith(".jpg"):
                    clean_path = os.path.join(clean_folder, filename)
                    noisy_path = os.path.join(input_folder, f"{noise}_{filename}")
                    denoised_path = os.path.join(output_folder, f"denoised_{noise}_{filename}")

                    try:
                        # Charger les images
                        clean_image = np.array(Image.open(clean_path).convert('RGB')) / 255.0
                        noisy_image = np.array(Image.open(noisy_path).convert('RGB')) / 255.0
                        denoised_image = np.array(Image.open(denoised_path).convert('RGB')) / 255.0

                        # Calculer PSNR et SSIM
                        psnr_values[noise].append(psnr(clean_image, denoised_image))
                        ssim_values[noise].append(ssim(clean_image, denoised_image))

                    except Exception as e:
                        print(f"Erreur avec {filename} pour le bruit {noise}: {e}")

        # Moyennes des métriques
        psnr_means = {noise: np.mean(values) for noise, values in psnr_values.items()}
        ssim_means = {noise: np.mean(values) for noise, values in ssim_values.items()}

        # Création d'un DataFrame pour les graphiques
        df = pd.DataFrame({
            "Type de bruit": noise_types,
            "PSNR": [psnr_means[noise] for noise in noise_types],
            "SSIM": [ssim_means[noise] for noise in noise_types]
        })

        # Tracer les graphiques
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].bar(df["Type de bruit"], df["PSNR"], color='skyblue')
        axes[0].set_title("PSNR par type de bruit")
        axes[0].set_ylabel("PSNR")
        axes[0].set_xlabel("Type de bruit")

        axes[1].bar(df["Type de bruit"], df["SSIM"], color='lightgreen')
        axes[1].set_title("SSIM par type de bruit")
        axes[1].set_ylabel("SSIM")
        axes[1].set_xlabel("Type de bruit")

        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Compilation des images et affichage des métriques
        fig, axes = plt.subplots(len(os.listdir(clean_folder)), 5, figsize=(20, 4 * len(os.listdir(clean_folder))))

        for i, filename in enumerate(os.listdir(clean_folder)):
            if filename.endswith(".jpg"):
                clean_path = os.path.join(clean_folder, filename)
                clean_image = Image.open(clean_path).convert('RGB')

                # Afficher l'image propre
                axes[i, 0].imshow(clean_image)
                axes[i, 0].set_title("Clean")
                axes[i, 0].axis('off')

                for j, noise in enumerate(['gauss', 'poisson', 'sap', 'speckle']):
                    denoised_path = os.path.join(output_folder, f"denoised_{noise}_{filename}")

                    try:
                        denoised_image = Image.open(denoised_path).convert('RGB')

                        #
                        psnr_value = psnr(np.array(clean_image) / 255.0, np.array(denoised_image) / 255.0)
                        print(f"{filename} - {noise}: PSNR = {psnr_value:.2f}")
                        axes[i, j + 1].imshow(denoised_image)
                        axes[i, j + 1].set_title(f"{noise}\nPSNR: {psnr_value:.2f}")
                        axes[i, j + 1].axis('off')

                    except Exception as e:
                        print(f"Erreur avec {filename} pour le bruit {noise}: {e}")

        plt.tight_layout()
        plt.savefig('image_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    else:
        print("Choix invalide!")
