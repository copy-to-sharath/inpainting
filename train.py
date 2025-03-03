import os
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision import transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import optuna
import cv2  # For robust text detection
import timm  # For Vision Transformer

###############################################
# Robust Text Detection Module using EAST
###############################################

class RobustTextDetector:
    def __init__(self, model_path='frozen_east_text_detection.pb', conf_threshold=0.5, nms_threshold=0.4):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"EAST model not found at {model_path}. Please download the frozen model.")
        self.net = cv2.dnn.readNet(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image):
        """
        Detects text regions in the given BGR image (numpy array).
        Returns a binary mask of shape (1, H, W) with text regions set to 1.
        """
        orig_h, orig_w = image.shape[:2]
        # EAST expects dimensions that are multiples of 32; here we use 320x320.
        new_w, new_h = (320, 320)
        blob = cv2.dnn.blobFromImage(image, 1.0, (new_w, new_h),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        scores, geometry = self.net.forward(layer_names)
        boxes, confidences = self.decode_predictions(scores, geometry, self.conf_threshold)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.conf_threshold, self.nms_threshold)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                box = boxes[i]
                vertices = cv2.boxPoints(box)
                vertices = np.int0(vertices)
                cv2.fillPoly(mask, [vertices], 255)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # Shape: (1, H, W)
        return mask

    def decode_predictions(self, scores, geometry, scoreThresh):
        numRows, numCols = scores.shape[2:4]
        boxes = []
        confidences = []
        for y in range(numRows):
            scoresData = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(numCols):
                score = scoresData[x]
                if score < scoreThresh:
                    continue
                offsetX, offsetY = x * 4.0, y * 4.0
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]
                centerX = offsetX + cos * x1_data[x] + sin * x2_data[x]
                centerY = offsetY - sin * x1_data[x] + cos * x2_data[x]
                box = ((centerX, centerY), (w, h), -angle * 180.0 / np.pi)
                boxes.append(box)
                confidences.append(float(score))
        return boxes, confidences

###############################################
# Auxiliary Functions for Style Loss
###############################################

def gram_matrix(features):
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

###############################################
# Inpainting Transformer Module with ViT Encoder Option
###############################################

class InpaintingTransformer(nn.Module):
    def __init__(self, img_channels=3, embed_dim=256, num_heads=8, num_layers=4, use_vit_encoder=False):
        super().__init__()
        self.use_vit_encoder = use_vit_encoder
        self.embed_dim = embed_dim

        if self.use_vit_encoder:
            # Use a pretrained ViT encoder (ImageNet initialization via timm)
            self.vit_encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            vit_embed_dim = self.vit_encoder.embed_dim  # typically 768 for vit_base_patch16_224
            if vit_embed_dim != embed_dim:
                self.project = nn.Linear(vit_embed_dim, embed_dim)
            else:
                self.project = nn.Identity()
            # For a 224x224 image with 16x16 patches, there are 14x14 = 196 patches.
            self.num_patches = 196
            # Learned query tokens for the transformer decoder.
            self.query_embed = nn.Parameter(torch.zeros(self.num_patches, embed_dim))
        else:
            # Original encoder: a simple convolution.
            self.conv_in = nn.Conv2d(img_channels, embed_dim, kernel_size=3, padding=1)

        # Common transformer module (we use the decoder if using ViT encoder,
        # otherwise a full transformer for autoencoding).
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers if self.use_vit_encoder else num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        # A convolutional decoder to project tokens back to image space.
        self.decoder_conv = nn.Conv2d(embed_dim, img_channels, kernel_size=3, padding=1)

    @staticmethod
    def build_2d_sincos_pos_embed(embed_dim, height, width):
        grid_w = np.arange(width, dtype=np.float32)
        grid_h = np.arange(height, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape(2, -1)
        pos_embed = np.zeros((height * width, embed_dim), dtype=np.float32)
        dim_half = embed_dim // 2
        div_term = np.exp(np.arange(0, dim_half, 2, dtype=np.float32) * -(np.log(10000.0) / dim_half))
        pos_embed[:, 0:dim_half:2] = np.sin(grid[1].T * div_term)
        pos_embed[:, 1:dim_half:2] = np.cos(grid[1].T * div_term)
        pos_embed[:, dim_half::2] = np.sin(grid[0].T * div_term)
        pos_embed[:, dim_half+1::2] = np.cos(grid[0].T * div_term)
        return torch.tensor(pos_embed, dtype=torch.float32)

    def forward(self, x, mask):
        # If using the ViT encoder, expect input images resized to 224x224.
        if self.use_vit_encoder:
            masked_images = x.clone()
            masked_images[mask.bool()] = 0.0
            # Pass the masked images through the ViT encoder.
            tokens = self.vit_encoder.forward_features(masked_images)  # (B, num_tokens, vit_embed_dim)
            tokens = self.project(tokens)  # (B, num_tokens, embed_dim)
            B = x.size(0)
            queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, num_patches, embed_dim)
            # Use transformer decoder with the encoder tokens as memory.
            decoded = self.transformer.decoder(tgt=queries, memory=tokens)  # (B, num_patches, embed_dim)
            # Reshape tokens into a spatial map (assume square grid).
            B, N, E = decoded.shape
            H = W = int(np.sqrt(N))
            decoded = decoded.transpose(1, 2).view(B, E, H, W)
            # Upsample to the original resolution.
            decoded = nn.functional.interpolate(decoded, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            out = self.decoder_conv(decoded)
            return out
        else:
            feat = self.conv_in(x)  # (B, embed_dim, H, W)
            B, E, H, W = feat.shape
            feat_flat = feat.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
            pos_embed = self.build_2d_sincos_pos_embed(self.embed_dim, H, W).to(feat_flat.device).unsqueeze(0)
            feat_flat = feat_flat + pos_embed
            transformed = self.transformer(src=feat_flat, tgt=feat_flat)  # (B, H*W, embed_dim)
            transformed = transformed.transpose(1, 2).view(B, E, H, W)
            out = self.decoder_conv(transformed)
            return out

###############################################
# Discriminator for Adversarial Loss
###############################################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)

###############################################
# Lightning Module: Segmentation + Inpainting + Losses
###############################################

class SegmentationInpaintingModule(pl.LightningModule):
    def __init__(self, lr=1e-4, threshold=0.1, embed_dim=256, num_heads=8, num_layers=4,
                 use_adversarial=False, lambda_perceptual=0.1, lambda_style=0.1, lambda_adv=0.001,
                 use_vit_encoder=True):
        super().__init__()
        self.save_hyperparameters()
        # Pretrained instance segmentation (frozen).
        self.segmentation_model = maskrcnn_resnet50_fpn(pretrained=True)
        self.segmentation_model.eval()
        for param in self.segmentation_model.parameters():
            param.requires_grad = False

        # Robust text detector.
        self.text_detector = RobustTextDetector(model_path='frozen_east_text_detection.pb')

        # Inpainting network with ViT encoder option.
        self.inpainter = InpaintingTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_vit_encoder=use_vit_encoder
        )
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Adversarial loss components.
        self.use_adversarial = use_adversarial
        if self.use_adversarial:
            self.discriminator = Discriminator(in_channels=3)

        # VGG for perceptual and style losses.
        self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, images, masks):
        """
        images: (B, 3, H, W)
        masks: instance segmentation masks (B, 1, H, W)
        Additionally computes text masks using the robust text detector.
        """
        B = images.size(0)
        text_masks = []
        for i in range(B):
            # Convert tensor (RGB) to numpy BGR for OpenCV.
            img_np = images[i].detach().cpu().mul(255).byte().permute(1, 2, 0).numpy()
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            text_mask_np = self.text_detector.detect(img_bgr)  # (1, H, W)
            text_mask_tensor = torch.from_numpy(text_mask_np).to(images.device)
            text_masks.append(text_mask_tensor)
        text_masks = torch.stack(text_masks, dim=0)  # (B, 1, H, W)
        # Combine instance segmentation mask and text mask.
        combined_masks = torch.max(masks, text_masks)
        masked_images = images.clone()
        masked_images[combined_masks.bool()] = 0.0
        inpainted = self.inpainter(masked_images, combined_masks)
        return inpainted

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        images, targets = batch
        images = torch.stack(images, dim=0)
        device = images.device

        # Combine instance segmentation masks.
        combined_inst_masks = []
        for t in targets:
            if "masks" in t and t["masks"].numel() > 0:
                combined = (t["masks"].sum(dim=0, keepdim=True) > 0).float()
            else:
                combined = torch.zeros(1, images.shape[2], images.shape[3], device=device)
            combined_inst_masks.append(combined)
        inst_masks = torch.stack(combined_inst_masks, dim=0)

        output = self.forward(images, inst_masks)
        loss_l1 = self.l1_loss(output * inst_masks, images * inst_masks)
        loss_mse = self.mse_loss(output * inst_masks, images * inst_masks)

        # Perceptual and style losses.
        feat_out = self.vgg(output)
        feat_gt = self.vgg(images)
        loss_perceptual = self.mse_loss(feat_out, feat_gt)
        loss_style = self.l1_loss(gram_matrix(feat_out), gram_matrix(feat_gt))

        if self.use_adversarial:
            if optimizer_idx == 0:
                adv_pred = self.discriminator(output)
                valid = torch.ones_like(adv_pred)
                loss_adv = nn.functional.binary_cross_entropy_with_logits(adv_pred, valid)
                loss_gen = (loss_l1 + loss_mse +
                            self.hparams.lambda_perceptual * loss_perceptual +
                            self.hparams.lambda_style * loss_style +
                            self.hparams.lambda_adv * loss_adv)
                self.log('train_gen_loss', loss_gen, prog_bar=True)
                return loss_gen
            elif optimizer_idx == 1:
                real_pred = self.discriminator(images)
                valid = torch.ones_like(real_pred)
                loss_real = nn.functional.binary_cross_entropy_with_logits(real_pred, valid)
                fake_pred = self.discriminator(output.detach())
                fake = torch.zeros_like(fake_pred)
                loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_pred, fake)
                loss_disc = (loss_real + loss_fake) / 2
                self.log('train_disc_loss', loss_disc, prog_bar=True)
                return loss_disc
        else:
            loss = loss_l1 + loss_mse + self.hparams.lambda_perceptual * loss_perceptual + self.hparams.lambda_style * loss_style
            self.log("train_loss", loss, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images, dim=0)
        device = images.device

        combined_inst_masks = []
        for t in targets:
            if "masks" in t and t["masks"].numel() > 0:
                combined = (t["masks"].sum(dim=0, keepdim=True) > 0).float()
            else:
                combined = torch.zeros(1, images.shape[2], images.shape[3], device=device)
            combined_inst_masks.append(combined)
        inst_masks = torch.stack(combined_inst_masks, dim=0)

        output = self.forward(images, inst_masks)
        loss_l1 = self.l1_loss(output * inst_masks, images * inst_masks)
        loss_mse = self.mse_loss(output * inst_masks, images * inst_masks)
        mse_val = loss_mse.item()
        psnr = 10 * math.log10(1.0 / (mse_val + 1e-8))
        f1 = self.compute_fbeta(output, images, inst_masks, beta=1)
        f2 = self.compute_fbeta(output, images, inst_masks, beta=2)

        self.log("val_l1_loss", loss_l1, prog_bar=True)
        self.log("val_mse_loss", loss_mse, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_f2", f2, prog_bar=True)
        return {"val_l1": loss_l1, "val_mse": loss_mse, "psnr": psnr, "f1": f1, "f2": f2}

    def configure_optimizers(self):
        if self.use_adversarial:
            optimizer_gen = optim.Adam(self.inpainter.parameters(), lr=self.hparams.lr)
            optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
            return [optimizer_gen, optimizer_disc], []
        else:
            optimizer = optim.Adam(self.inpainter.parameters(), lr=self.hparams.lr)
            return optimizer

    def compute_fbeta(self, output, target, mask, beta=1.0):
        error = torch.abs(output - target)
        pred = (error < self.hparams.threshold).float()
        tp = (pred * mask).sum()
        total = mask.sum()
        precision = tp / (total + 1e-8)
        recall = precision
        fbeta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + 1e-8)
        return fbeta

###############################################
# COCO Dataset and Data Module
###############################################

class CocoSegmentationDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.root = root
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        masks = []
        for ann in anns:
            if "segmentation" in ann:
                mask = coco.annToMask(ann)
                mask = torch.as_tensor(mask, dtype=torch.float32)
                masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, image.shape[1], image.shape[2]))
        target = {"masks": masks}
        return image, target

    def __len__(self):
        return len(self.ids)

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, ann_file, batch_size=4, num_workers=4, val_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        # If using ViT encoder, resize images to 224x224; otherwise, 256x256.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        full_dataset = CocoSegmentationDataset(root=self.data_dir,
                                                annFile=self.ann_file,
                                                transform=self.transform)
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        return images, targets

###############################################
# Functions for Saving the Model
###############################################

def save_huggingface_format(model: SegmentationInpaintingModule, save_dir="hf_model"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    config = model.hparams.copy() if hasattr(model, "hparams") else {}
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"Model saved in Hugging Face format at {save_dir}")

def save_torchserve_mar(model: SegmentationInpaintingModule,
                        model_name="inpainting_model",
                        version="1.0",
                        export_path="model_store",
                        handler="handler.py",
                        extra_files="config.json"):
    os.makedirs("tmp_model", exist_ok=True)
    model_file = os.path.join("tmp_model", "model.pth")
    torch.save(model.state_dict(), model_file)
    os.makedirs(export_path, exist_ok=True)
    cmd = (
        f"torch-model-archiver --model-name {model_name} "
        f"--version {version} --serialized-file {model_file} "
        f"--handler {handler} --extra-files {extra_files} --export-path {export_path}"
    )
    print("Packaging model for TorchServe with command:")
    print(cmd)
    os.system(cmd)
    print(f"Model archive saved in {export_path}")

###############################################
# Hyperparameter Tuning with Optuna
###############################################

def objective(trial, args):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    embed_dim = trial.suggest_categorical("embed_dim", [256, 384])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    threshold = trial.suggest_uniform("threshold", 0.05, 0.2)
    use_adv = trial.suggest_categorical("use_adversarial", [False, True])
    lambda_perc = trial.suggest_uniform("lambda_perceptual", 0.01, 0.2)
    lambda_st = trial.suggest_uniform("lambda_style", 0.01, 0.2)
    lambda_adv = trial.suggest_uniform("lambda_adv", 0.0001, 0.01)
    use_vit = trial.suggest_categorical("use_vit_encoder", [False, True])

    model = SegmentationInpaintingModule(
        lr=lr,
        threshold=threshold,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        use_adversarial=use_adv,
        lambda_perceptual=lambda_perc,
        lambda_style=lambda_st,
        lambda_adv=lambda_adv,
        use_vit_encoder=use_vit
    )
    dm = CocoDataModule(data_dir=args.data_dir, ann_file=args.ann_file,
                        batch_size=args.batch_size, num_workers=args.num_workers, val_split=0.1)
    dm.setup()
    trainer = Trainer(
        max_epochs=3,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=False,
        checkpoint_callback=False
    )
    trainer.fit(model, dm)
    val_loss = trainer.callback_metrics.get("val_l1_loss")
    return val_loss.item() if val_loss is not None else float("inf")

###############################################
# Main: Training and Optional Tuning with Resume Support
###############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COCO images (e.g., train2017 folder)")
    parser.add_argument("--ann_file", type=str, required=True, help="Path to COCO annotations JSON file")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args), n_trials=10)
        print("Best hyperparameters:")
        print(study.best_trial.params)
    else:
        model = SegmentationInpaintingModule(lr=1e-4)
        dm = CocoDataModule(data_dir=args.data_dir, ann_file=args.ann_file,
                            batch_size=args.batch_size, num_workers=args.num_workers, val_split=0.1)
        trainer = Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=args.max_epochs,
            resume_from_checkpoint=args.resume_checkpoint
        )
        trainer.fit(model, dm)
        save_huggingface_format(model, save_dir="hf_model")
        # Uncomment the following line if you have a custom TorchServe handler.
        # save_torchserve_mar(model, handler="handler.py")
