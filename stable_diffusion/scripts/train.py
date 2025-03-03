# scripts/train.py
import os
import yaml
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import torch
from lightning.lightning_module import CocoTransformerModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import argparse
import random
import numpy as np

# Dummy dataset for demonstration – replace with actual COCO dataset loader.
class DummyCocoDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=80):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return dummy image and random label
        image = torch.randn(3, 224, 224)
        label = random.randint(0, self.num_classes - 1)
        return image, label

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(config):
    # Loop over multiple seeds for multiple initializations.
    for seed in config.experiment.seed:
        set_seed(seed)
        print(f"Training with seed: {seed}")
        
        # Initialize MLflow run if enabled.
        if config.logging.mlflow:
            mlflow.start_run(run_name=config.experiment.name + f"_seed_{seed}")
            mlflow.log_params(OmegaConf.to_container(config, resolve=True))
            
        # Initialize TensorBoard logger if enabled.
        tb_logger = TensorBoardLogger("tb_logs", name=config.experiment.name)
        
        # Create dataset and dataloaders.
        train_dataset = DummyCocoDataset(num_samples=500, num_classes=config.model.num_classes)
        val_dataset = DummyCocoDataset(num_samples=100, num_classes=config.model.num_classes)
        train_loader = DataLoader(train_dataset, batch_size=config.experiment.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.experiment.batch_size)
        
        # Create Lightning Module.
        model = CocoTransformerModule(config)
        
        # Model checkpoint callback.
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.paths.model_save_dir,
            filename=f"{config.experiment.name}_seed_{seed}" + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        
        # Trainer (using mixed precision for GPU memory optimization)
        trainer = pl.Trainer(
            max_epochs=config.experiment.max_epochs,
            precision=config.training.precision,
            gpus=1 if torch.cuda.is_available() else 0,
            logger=tb_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Log final metrics to MLflow.
        metrics_dict = trainer.callback_metrics
        if config.logging.mlflow:
            mlflow.log_metrics({k: float(v.cpu().numpy()) if torch.is_tensor(v) else v 
                                for k, v in metrics_dict.items()})
            mlflow.end_run()
        
        # Save the model in both formats.
        from utils import utils
        os.makedirs(config.paths.model_save_dir, exist_ok=True)
        mar_path = os.path.join(config.paths.model_save_dir, f"model_seed_{seed}.mar")
        hf_path = os.path.join(config.paths.model_save_dir, f"hf_model_seed_{seed}")
        utils.save_model_pytorch_mar(model.model, file_path=mar_path)
        utils.save_model_huggingface(model.model, file_path=hf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # Convert dict to OmegaConf for dot-access
    config = OmegaConf.create(config)
    main(config)
