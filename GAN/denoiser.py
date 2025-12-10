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


# ============= FONCTION DE DÉCOUPAGE EN PATCHS =============
def extract_patches(image_tensor, patch_size=125, overlap=25):
    """
    Découpe une image en patchs avec chevauchement
    
    Args:
        image_tensor: Tensor de forme [C, H, W]
        patch_size: Taille des patchs (carré)
        overlap: Taille du chevauchement entre patchs
    
    Returns:
        patches: Liste de tensors [C, patch_size, patch_size]
        positions: Liste des positions (y, x) de chaque patch
        original_shape: Dimensions originales (H, W)
    """
    C, H, W = image_tensor.shape
    stride = patch_size - overlap
    
    patches = []
    positions = []
    
    # Parcourir l'image avec chevauchement
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Ajuster la dernière ligne/colonne si nécessaire
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            y_start = max(0, y_end - patch_size)
            x_start = max(0, x_end - patch_size)
            
            patch = image_tensor[:, y_start:y_end, x_start:x_end]
            
            # S'assurer que le patch a la bonne taille
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                # Padding si nécessaire
                pad_h = patch_size - patch.shape[1]
                pad_w = patch_size - patch.shape[2]
                patch = nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
            
            patches.append(patch)
            positions.append((y_start, x_start))
    
    return patches, positions, (H, W)


def reconstruct_from_patches(denoised_patches, positions, original_shape, patch_size=125, overlap=25):
    """
    Reconstruit une image à partir de patchs débruités avec moyennage dans les zones de chevauchement
    
    Args:
        denoised_patches: Liste de tensors débruités [C, patch_size, patch_size]
        positions: Liste des positions (y, x) de chaque patch
        original_shape: Dimensions originales (H, W)
        patch_size: Taille des patchs
        overlap: Taille du chevauchement
    
    Returns:
        Tensor reconstruit [C, H, W]
    """
    H, W = original_shape
    C = denoised_patches[0].shape[0]
    
    # Images pour accumuler les valeurs et compter les contributions
    reconstructed = torch.zeros((C, H, W))
    weight_map = torch.zeros((H, W))
    
    # Créer un masque de pondération pour le chevauchement (fenêtre de Hann)
    window = torch.ones((patch_size, patch_size))
    if overlap > 0:
        fade = torch.linspace(0, 1, overlap)
        # Bords gauche et droit
        window[:, :overlap] *= fade.unsqueeze(0)
        window[:, -overlap:] *= fade.flip(0).unsqueeze(0)
        # Bords haut et bas
        window[:overlap, :] *= fade.unsqueeze(1)
        window[-overlap:, :] *= fade.flip(0).unsqueeze(1)
    
    # Placer chaque patch à sa position avec pondération
    for patch, (y, x) in zip(denoised_patches, positions):
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        
        # Extraire la partie valide du patch
        valid_h = y_end - y
        valid_w = x_end - x
        valid_patch = patch[:, :valid_h, :valid_w]
        valid_window = window[:valid_h, :valid_w]
        
        # Ajouter avec pondération
        reconstructed[:, y:y_end, x:x_end] += valid_patch * valid_window
        weight_map[y:y_end, x:x_end] += valid_window
    
    # Normaliser par les poids
    reconstructed = reconstructed / weight_map.clamp(min=1e-8)
    
    return reconstructed


# ============= DÉBRUITAGE PAR PATCHS =============
def denoise_by_patches(image_tensor, model, device, patch_size=125, overlap=25):
    """
    Débruite une image par patchs avec chevauchement
    
    Args:
        image_tensor: Tensor [1, C, H, W]
        model: Modèle de débruitage
        device: Device PyTorch
        patch_size: Taille des patchs
        overlap: Chevauchement entre patchs
    
    Returns:
        Tensor débruité [1, C, H, W]
    """
    # Extraire le tensor sans batch dimension
    img = image_tensor.squeeze(0)
    
    # Découper en patchs
    print(f"Découpage en patchs {patch_size}x{patch_size} avec chevauchement de {overlap}px...")
    patches, positions, original_shape = extract_patches(img, patch_size, overlap)
    print(f"Nombre de patchs: {len(patches)}")
    
    # Débruiter chaque patch
    denoised_patches = []
    print("Débruitage des patchs...")
    for i, patch in enumerate(patches):
        if (i + 1) % 10 == 0:
            print(f"  Patch {i+1}/{len(patches)}")
        
        patch_batch = patch.unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_patch = model(patch_batch).cpu().squeeze(0)
        denoised_patches.append(denoised_patch)
    
    # Reconstruire l'image
    print("Reconstruction de l'image...")
    reconstructed = reconstruct_from_patches(denoised_patches, positions, original_shape, patch_size, overlap)
    
    return reconstructed.unsqueeze(0)


# ============= DEBRUITAGE IMAGE COMPLETE =============
def denoise_full_image(image_tensor, model, device):
    """
    Débruite l'image complète en une seule fois
    
    Args:
        image_tensor: Tensor [1, C, H, W]
        model: Modèle de débruitage
        device: Device PyTorch
    
    Returns:
        Tensor débruité [1, C, H, W]
    """
    print("Débruitage de l'image complète...")
    with torch.no_grad():
        denoised = model(image_tensor.to(device)).cpu()
    return denoised


# ============= SCRIPT PRINCIPAL : Débruiter une image =============
def denoise_image(image_path, model_path="gan.pth", save_path=None, method="patches"):
    """
    Applique le modèle de débruitage sur une image
    
    Args:
        image_path: Chemin vers l'image à débruiter
        model_path: Chemin vers le fichier .pth du modèle
        save_path: Chemin pour sauvegarder l'image débruitée (optionnel)
        method: "patches" (par patchs) ou "full" (image complète)
    """
    # Device
    use_directml = False
    try:
        import torch_directml
        device = torch_directml.device()
        use_directml = True
        print("Utilisation de DirectML")
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = UNet(in_channels=3, out_channels=3)
    
    # Charger les poids avec la bonne méthode selon le device
    if use_directml:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extraire les poids du modèle selon le format du checkpoint
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        print(f"Checkpoint détecté (epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Modèle chargé\n")
    
    # Charger et préparer l'image
    print(f"Chargement de l'image {image_path}...")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"Taille de l'image: {original_size[0]}x{original_size[1]}\n")
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Débruitage selon la méthode choisie
    if method == "patches":
        denoised_tensor = denoise_by_patches(image_tensor, model, device, patch_size=125, overlap=25)
    else:
        denoised_tensor = denoise_full_image(image_tensor, model, device)
    
    # Convertir en image
    denoised_tensor = denoised_tensor.squeeze(0)
    denoised_array = denoised_tensor.permute(1, 2, 0).numpy()
    denoised_array = np.clip(denoised_array, 0, 1)
    denoised_image = Image.fromarray((denoised_array * 255).astype(np.uint8))
    
    # Sauvegarder si demandé
    if save_path:
        denoised_image.save(save_path)
        print(f"\nImage débruitée sauvegardée dans {save_path}")
    
    # Afficher la comparaison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Image bruitée (originale)')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_image)
    axes[1].set_title(f'Image débruitée ({method})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nTerminé !")
    return denoised_image


# ============= DEBRUITAGE D'UN DOSSIER =============
def denoise_folder(input_folder, output_folder, model_path="gan.pth", method="patches"):
    """
    Débruite toutes les images d'un dossier
    
    Args:
        input_folder: Dossier contenant les images bruitées
        output_folder: Dossier où sauvegarder les images débruitées
        model_path: Chemin vers le fichier .pth du modèle
        method: "patches" (par patchs) ou "full" (image complète)
    """
    import os
    from pathlib import Path
    
    # Créer le dossier de sortie s'il n'existe pas
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Device
    use_directml = False
    try:
        import torch_directml
        device = torch_directml.device()
        use_directml = True
        print("Utilisation de DirectML")
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = UNet(in_channels=3, out_channels=3)
    
    # Charger les poids avec la bonne méthode selon le device
    if use_directml:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extraire les poids du modèle selon le format du checkpoint
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        print(f"Checkpoint détecté (epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Modèle chargé!\n")
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Lister les images
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Traitement de {len(image_files)} images avec la méthode '{method}'...\n")
    
    # Traiter chaque image
    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"denoised_{filename}")
        
        try:
            print(f"\n[{idx}/{len(image_files)}] Traitement de {filename}...")
            
            # Charger l'image
            image = Image.open(input_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # Débruiter selon la méthode
            if method == "patches":
                denoised_tensor = denoise_by_patches(image_tensor, model, device, patch_size=125, overlap=25)
            else:
                denoised_tensor = denoise_full_image(image_tensor, model, device)
            
            # Convertir et sauvegarder
            denoised_tensor = denoised_tensor.squeeze(0)
            denoised_array = denoised_tensor.permute(1, 2, 0).numpy()
            denoised_array = np.clip(denoised_array, 0, 1)
            denoised_image = Image.fromarray((denoised_array * 255).astype(np.uint8))
            denoised_image.save(output_path)
            
            print(f"✓ Sauvegardé: {output_path}")
            
        except Exception as e:
            print(f"✗ Erreur avec {filename}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Traitement terminé ! Images sauvegardées dans {output_folder}")
    print(f"{'='*60}")


# ============= EXEMPLES D'UTILISATION =============
if __name__ == "__main__":
    print("=" * 60)
    print("SCRIPTS DE DÉBRUITAGE")
    print("=" * 60)
    print("1. Débruiter une seule image")
    print("2. Débruiter un dossier entier")
    print("=" * 60)
    
    choice = input("\nVotre choix (1/2): ")
    
    if choice == "1":
        print("\n" + "=" * 60)
        print("DÉBRUITAGE D'UNE IMAGE")
        print("=" * 60)
        
        # Demander la méthode
        print("\nMéthode de débruitage:")
        print("1. Par patchs 125x125")
        print("2. Image complète")
        method_choice = input("Votre choix (1/2): ")
        method = "patches" if method_choice == "1" else "full"
        
        print(f"\n→ Méthode sélectionnée: {'Par patchs' if method == 'patches' else 'Image complète'}\n")
        
        image_path = input("Chemin de l'image à débruiter: ")
        save_path = input("Chemin de sauvegarde (Entrée pour ne pas sauvegarder): ")
        save_path = save_path if save_path else None
        
        denoise_image(image_path, save_path=save_path, method=method)
        
    elif choice == "2":
        print("\n" + "=" * 60)
        print("DÉBRUITAGE D'UN DOSSIER")
        print("=" * 60)
        
        # Demander la méthode
        print("\nMéthode de débruitage:")
        print("1. Par patchs 125x125")
        print("2. Image complète")
        method_choice = input("Votre choix (1/2): ")
        method = "patches" if method_choice == "1" else "full"
        
        print(f"\n→ Méthode sélectionnée: {'Par patchs' if method == 'patches' else 'Image complète'}\n")
        
        input_folder = input("Dossier contenant les images bruitées: ")
        output_folder = input("Dossier de sortie pour les images débruitées: ")
        
        denoise_folder(input_folder, output_folder, method=method)
        
    else:
        print("Choix invalide!")