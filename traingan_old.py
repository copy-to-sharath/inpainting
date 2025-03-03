import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

# Define the Generator and Discriminator networks
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=13):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 64 * 64),  # Output image size 64x64
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, noise, labels):
        embedded_labels = self.label_embedding(labels)
        gen_input = torch.cat((noise, embedded_labels), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 3, 64, 64)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        embedded_labels = self.label_embedding(labels)
        d_in = torch.cat((img.view(img.size(0), -1), embedded_labels), -1)
        validity = self.model(d_in)
        return validity

class CocoGAN(pl.LightningModule):
    def __init__(self, latent_dim=100, learning_rate=0.0002, num_classes=13):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim, num_classes)
        self.discriminator = Discriminator(num_classes)

        self.criterion = nn.BCELoss()

    def forward(self, z, labels):
        return self.generator(z, labels)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, targets = batch
        labels = torch.tensor([target['category_id'][0] if target['category_id'] else 0 for target in targets], device=self.device) #get first category ID, or 0 if empty
        labels = labels-1 # COCO category IDs start at 1, shift to 0-12
        batch_size = imgs.size(0)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Train Generator
        if optimizer_idx == 0:
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            gen_imgs = self(z, labels)
            g_loss = self.criterion(self.discriminator(gen_imgs, labels), valid)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # Train Discriminator
        if optimizer_idx == 1:
            real_loss = self.criterion(self.discriminator(imgs, labels), valid)
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            gen_imgs = self(z, labels).detach()
            fake_loss = self.criterion(self.discriminator(gen_imgs, labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        g_optim = optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate)
        d_optim = optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate)
        return [g_optim, d_optim], []

def prepare_coco_data(image_dir, annotation_file, batch_size=64, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CocoDetection(root=image_dir, annFile=annotation_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    return dataloader

def train_coco_gan(image_dir, annotation_file, latent_dim=100, max_epochs=50, batch_size=64, image_size=64, num_classes=13):
    dataloader = prepare_coco_data(image_dir, annotation_file, batch_size, image_size)
    model = CocoGAN(latent_dim=latent_dim, num_classes=num_classes)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, dataloader)
    return model

if __name__ == "__main__":
    image_dir = '/path/to/your/coco/images/train2017'
    annotation_file = '/path/to/your/coco/annotations/instances_train2017.json'

    trained_gan = train_coco_gan(image_dir, annotation_file)
    print("GAN training complete.")