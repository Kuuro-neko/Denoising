import os
import shutil

def keep_first_n_images(folder_path, n):
    """Keep only the first n images in the flickr_images folder."""
    # Get all files in the folder
    files = sorted(os.listdir(folder_path))
    
    # Filter for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Remove images beyond the first n
    for img in images[n:]:
        file_path = os.path.join(folder_path, img)
        os.remove(file_path)
    
    print(f"Kept {min(n, len(images))} images out of {len(images)}")

if __name__ == "__main__":
    folder = "flickr_images"
    n = 5000
    keep_first_n_images(folder, n)