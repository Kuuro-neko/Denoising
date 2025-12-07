import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

class ImagePatchGenerator:
    def __init__(self, input_dir, output_dir, patch_size=125):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        
        # Créer les dossiers de sortie
        self.dirs = {
            'clean': self.output_dir,
            'gaussian': self.output_dir,
            'poisson': self.output_dir,
            'salt_pepper': self.output_dir,
            'speckle': self.output_dir
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def add_gaussian_noise(self, image, mean=0, std=25):
        """Ajoute du bruit gaussien"""
        noise = np.random.normal(mean, std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_poisson_noise(self, image):
        """Ajoute du bruit de Poisson"""
        # Normaliser à [0,1], appliquer Poisson, puis revenir à [0,255]
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image / 255.0 * vals) / vals * 255.0
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        """Ajoute du bruit poivre et sel"""
        noisy = image.copy()
        
        # Sel (pixels blancs)
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        noisy[salt_mask] = 255
        
        # Poivre (pixels noirs)
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        noisy[pepper_mask] = 0
        
        return noisy
    
    def add_speckle_noise(self, image, std=0.1):
        """Ajoute du bruit speckle (multiplicatif)"""
        noise = np.random.randn(*image.shape) * std
        noisy = image + image * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def extract_patches(self, image):
        """Découpe l'image en patchs de taille fixe sans chevauchement"""
        h, w = image.shape[:2]
        
        # Nombre de patchs dans chaque dimension
        n_patches_h = h // self.patch_size
        n_patches_w = w // self.patch_size
        
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y = i * self.patch_size
                x = j * self.patch_size
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
        
        return patches
    
    def process_image(self, image_path, start_index):
        """Traite une seule image"""
        # Lire l'image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Erreur de lecture: {image_path}")
            return 0
        
        # Extraire les patchs
        patches = self.extract_patches(image)
        
        # Pour chaque patch
        for idx, patch in enumerate(patches):
            patch_number = start_index + idx
            
            # Nom pour le patch original
            clean_name = f"{patch_number:06d}.jpg"
            cv2.imwrite(str(self.dirs['clean'] / clean_name), patch)
            
            # Générer et sauvegarder les versions bruitées
            cv2.imwrite(
                str(self.dirs['gaussian'] / f"gauss_{patch_number:06d}.jpg"),
                self.add_gaussian_noise(patch)
            )

            cv2.imwrite(
                str(self.dirs['gaussian'] / f"gauss_weak_{patch_number:06d}.jpg"),
                self.add_gaussian_noise(patch, std=10)
            )

            cv2.imwrite(
                str(self.dirs['gaussian'] / f"gauss_strong_{patch_number:06d}.jpg"),
                self.add_gaussian_noise(patch, std=50)
            )
            
            cv2.imwrite(
                str(self.dirs['poisson'] / f"poisson_{patch_number:06d}.jpg"),
                self.add_poisson_noise(patch)
            )
            
            cv2.imwrite(
                str(self.dirs['salt_pepper'] / f"sap_{patch_number:06d}.jpg"),
                self.add_salt_pepper_noise(patch)
            )
            
            cv2.imwrite(
                str(self.dirs['speckle'] / f"speckle_{patch_number:06d}.jpg"),
                self.add_speckle_noise(patch)
            )
        
        return len(patches)
    
    def process_dataset(self):
        """Traite toutes les images du dossier d'entrée"""
        # Liste des images
        image_files = list(self.input_dir.glob('*.jpg')) + \
                      list(self.input_dir.glob('*.jpeg')) + \
                      list(self.input_dir.glob('*.png'))
        
        print(f"Nombre d'images trouvées: {len(image_files)}")
        
        total_patches = 0
        current_index = 1  # Commence la numérotation à 1
        
        # Traiter chaque image avec barre de progression
        for image_path in tqdm(image_files, desc="Traitement des images"):
            n_patches = self.process_image(image_path, current_index)
            total_patches += n_patches
            current_index += n_patches
        
        print(f"\nTraitement terminé!")
        print(f"Total de patchs générés: {total_patches}")
        print(f"Total d'images générées: {total_patches * 5} (clean + 4 bruits)")
        print(f"\nStructure des dossiers:")
        for noise_type, path in self.dirs.items():
            print(f"  {noise_type}: {path}")


def main():
    # Configuration
    INPUT_DIR = "./flickr30k_images/flickr30k_images"  # Dossier contenant les images Flickr
    OUTPUT_DIR = "./patches"        # Dossier de sortie pour les patchs
    PATCH_SIZE = 125
    
    # Créer le générateur et traiter les images
    generator = ImagePatchGenerator(INPUT_DIR, OUTPUT_DIR, PATCH_SIZE)
    generator.process_dataset()


if __name__ == "__main__":
    main()