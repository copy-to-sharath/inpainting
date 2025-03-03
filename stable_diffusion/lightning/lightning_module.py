# lightning/lightning_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.transformer_model import TransformerClassifier
from utils import metrics

class CocoTransformerModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = TransformerClassifier(
            norm_type=self.hparams.model.norm_type,
            num_classes=self.hparams.model.num_classes
        )
        self.learning_rate = self.hparams.experiment.learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # assume batch returns images and labels
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "logits": logits, "y": y}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        
        # Compute additional metrics
        acc = metrics.compute_accuracy(y, logits)
        f1 = metrics.compute_f1(y, logits)
        f2 = metrics.compute_f2(y, logits)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_f2", f2, prog_bar=True)
        return {"val_loss": loss, "acc": acc, "f1": f1, "f2": f2}
    
    def validation_epoch_end(self, outputs):
        # Optionally combine train and val loss if needed
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_val_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
