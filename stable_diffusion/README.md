# Synthetic COCO Training with Diffusion & Transformers

This project demonstrates generating synthetic data using Stable Diffusion and ControlNet, then training a transformer‑based model on the COCO dataset using PyTorch Lightning. Hyperparameter tuning, logging with MLflow & TensorBoard, and model export in both PyTorch MAR and Hugging Face formats are included.

## Folder Structure


## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt

python scripts/train.py --config configs/config.yaml
