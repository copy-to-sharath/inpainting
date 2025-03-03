# scripts/inference.py
import torch
from lightning.lightning_module import CocoTransformerModule
from PIL import Image
import argparse
from torchvision import transforms

def load_model(model_path):
    model = CocoTransformerModule.load_from_checkpoint(model_path)
    model.eval()
    return model

def predict(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    pred = torch.argmax(logits, dim=1)
    return pred.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    prediction = predict(model, args.image)
    print(f"Predicted class: {prediction}")
