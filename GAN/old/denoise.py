import torch
from torchvision import transforms
from PIL import Image
from GAN import UNet  # Assuming UNet is defined in GAN.py

def denoise_image(image_path, model_path, output_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained generator model
    generator = UNet().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    noisy_image = transform(image).unsqueeze(0).to(device)

    # Denoise the image
    with torch.no_grad():
        denoised_image = generator(noisy_image)

    # Convert the output tensor to a PIL image
    denoised_image = denoised_image.squeeze(0).cpu().clamp(0, 1)
    denoised_image = transforms.ToPILImage()(denoised_image)

    # Save the denoised image
    denoised_image.save(output_path)
    print(f"Denoised image saved to {output_path}")

if __name__ == "__main__":
    # Paths
    image_path = "./image_database/patches/000001.jpg"  # Path to the noisy input image
    model_path = "./gan.pth"  # Path to the saved generator model
    output_path = "denoised_image.jpg"  # Path to save the denoised image

    # Denoise the image
    denoise_image(image_path, model_path, output_path)