# synthetic_data/stable_diffusion.py
from diffusers import StableDiffusionPipeline
import torch

def generate_synthetic_images(prompt: str, num_images: int = 5, device: str = "cuda"):
    """
    Generate synthetic images using Stable Diffusion.
    Note: You must have your Hugging Face token and model downloaded.
    """
    # Load the Stable Diffusion pipeline (ensure you have downloaded the model)
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to(device)
    
    images = []
    for _ in range(num_images):
        image = pipe(prompt).images[0]
        images.append(image)
    return images

if __name__ == "__main__":
    # Example usage:
    imgs = generate_synthetic_images("A photo of a cat on a sunny day", num_images=3)
    for idx, img in enumerate(imgs):
        img.save(f"synthetic_cat_{idx}.png")
