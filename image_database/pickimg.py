import os
import shutil
import glob

def process_images(input_folder, output_folder, image_name):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    patterns = [
        f"gauss_{image_name}",
        f"poisson_{image_name}",
        f"sap_{image_name}",
        f"speckle_{image_name}",
        image_name
    ]

    print("Processing image:", image_name)
    print("Looking for patterns:", patterns)

    
    for pattern in patterns:
        search_pattern = os.path.join(input_folder, pattern)
        for file_path in glob.glob(search_pattern):
            shutil.copy(file_path, output_folder)
            print(f"Copied {file_path} to {output_folder}")

# Example usage
if __name__ == "__main__":
    input_folder = "./patches"
    output_folder = "./output"
    image_name = "007055.jpg"
    
    process_images(input_folder, output_folder, image_name)