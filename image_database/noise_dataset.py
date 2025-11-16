import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random


class NoiseApplicator:
    def __init__(self, input_dir, output_dir_clean, output_dir_noisy, limit=None):
        self.input_dir = Path(input_dir)
        self.output_dir_clean = Path(output_dir_clean)
        self.output_dir_noisy = Path(output_dir_noisy)
        self.limit = limit
        
        # Create output directories
        self.output_dir_clean.mkdir(parents=True, exist_ok=True)
        self.output_dir_noisy.mkdir(parents=True, exist_ok=True)
    
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
    
    def apply_random_noise(self, image):
        """Apply one of the 4 noise types randomly"""
        noise_functions = [
            self.add_gaussian_noise,
            self.add_poisson_noise,
            self.add_salt_pepper_noise,
            self.add_speckle_noise
        ]
        
        noise_func = random.choice(noise_functions)
        return noise_func(image)
    
    def process_image(self, image_path):
        """Process a single image: resize to 64x64, save clean version, apply noise, save noisy version"""
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error reading: {image_path}")
            return False
        
        # Resize to 64x64
        resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Save clean 64x64 image
        filename = image_path.name
        clean_output_path = self.output_dir_clean / filename
        cv2.imwrite(str(clean_output_path), resized_image)
        
        # Apply random noise to the resized image
        noisy_image = self.apply_random_noise(resized_image)
        
        # Create output filename: example.jpg -> noisy_example.jpg
        noisy_filename = f"noisy_{filename}"
        noisy_output_path = self.output_dir_noisy / noisy_filename
        
        # Save noisy image
        cv2.imwrite(str(noisy_output_path), noisy_image)
        return True
    
    def process_dataset(self):
        """Process all images in the input directory"""
        # List all image files
        image_files = list(self.input_dir.glob('*.jpg')) + \
                      list(self.input_dir.glob('*.jpeg')) + \
                      list(self.input_dir.glob('*.png'))
        
        # Apply limit if specified
        if self.limit is not None and self.limit > 0:
            image_files = image_files[:self.limit]
            print(f"Processing limit set to {self.limit} images")
        
        print(f"Found {len(image_files)} images in {self.input_dir}")
        
        successful = 0
        # Process each image with progress bar
        for image_path in tqdm(image_files, desc="Applying noise to images"):
            if self.process_image(image_path):
                successful += 1
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful}/{len(image_files)} images")
        print(f"Clean output directory: {self.output_dir_clean}")
        print(f"Noisy output directory: {self.output_dir_noisy}")


def main():
    # Configuration
    INPUT_DIR = "./patches/clean"
    OUTPUT_DIR_CLEAN = "./patchesx64/clean"
    OUTPUT_DIR_NOISY = "./patchesx64/noisy"
    LIMIT = 25000  # Set to a number to limit images processed, or None to process all
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Create noise applicator and process images
    applicator = NoiseApplicator(INPUT_DIR, OUTPUT_DIR_CLEAN, OUTPUT_DIR_NOISY, limit=LIMIT)
    applicator.process_dataset()


if __name__ == "__main__":
    main()
