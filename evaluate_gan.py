import torch
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have the trained CocoGAN model from the previous code
# Load the trained model
def load_trained_model(model_path, latent_dim=100, num_classes=13):
    model = CocoGAN.load_from_checkpoint(model_path, latent_dim=latent_dim, num_classes=num_classes)
    model.eval()
    return model

# Inference script
def generate_images(model, num_images=10, latent_dim=100, num_classes=13):
    """Generates images from the trained GAN."""
    model.eval()
    generated_images = []
    with torch.no_grad():
        for _ in range(num_images):
            label = torch.randint(0, num_classes, (1,)) #random label
            z = torch.randn(1, latent_dim)
            generated_image = model(z, label)
            generated_images.append((generated_image.squeeze(0), label.item())) #save image and label.

    return generated_images

def show_generated_images(generated_images):
    """Displays generated images."""
    for img, label in generated_images:
        img_np = img.cpu().numpy()
        img_np = (img_np * 0.5) + 0.5 #unnormalize
        img_np = np.transpose(img_np, (1, 2, 0))
        plt.imshow(img_np)
        plt.title(f"Generated Image (Class: {label+1})")
        plt.show()

# Evaluation (simple visual inspection)
def evaluate_gan(model, image_dir, annotation_file, num_eval_images=5):
    """Performs a simple visual evaluation of the GAN."""
    dataloader = prepare_coco_data(image_dir, annotation_file, batch_size=1, image_size=64) #batch size 1 for easy display.
    model.eval()
    with torch.no_grad():
        for i, (real_img, targets) in enumerate(dataloader):
            if i >= num_eval_images:
                break
            label = torch.tensor([target['category_id'][0] if target['category_id'] else 0 for target in targets])
            label = label-1
            z = torch.randn(1, 100)
            generated_img = model(z, label)

            real_img_np = real_img.squeeze(0).cpu().numpy()
            real_img_np = (real_img_np * 0.5) + 0.5
            real_img_np = np.transpose(real_img_np, (1, 2, 0))

            generated_img_np = generated_img.squeeze(0).cpu().numpy()
            generated_img_np = (generated_img_np * 0.5) + 0.5
            generated_img_np = np.transpose(generated_img_np, (1, 2, 0))

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(real_img_np)
            plt.title(f"Real Image (Class: {label.item()+1})")

            plt.subplot(1, 2, 2)
            plt.imshow(generated_img_np)
            plt.title(f"Generated Image (Class: {label.item()+1})")
            plt.show()

# Data Loading and Preprocessing with CocoDetection (same as training)
def prepare_coco_data(image_dir, annotation_file, batch_size=64, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CocoDetection(root=image_dir, annFile=annotation_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    return dataloader

# Example Usage
if __name__ == "__main__":
    model_path = "path/to/your/trained_model.ckpt" #replace with model path
    image_dir = '/path/to/your/coco/images/train2017'
    annotation_file = '/path/to/your/coco/annotations/instances_train2017.json'

    if os.path.exists(model_path):
        trained_model = load_trained_model(model_path)

        # Inference: Generate and show images
        generated_images = generate_images(trained_model)
        show_generated_images(generated_images)

        # Evaluation: Compare real and generated images
        evaluate_gan(trained_model, image_dir, annotation_file)
    else:
        print(f"Error: Model file not found at {model_path}")