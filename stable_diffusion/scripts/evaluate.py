# scripts/evaluate.py
import torch
from lightning.lightning_module import CocoTransformerModule
from torch.utils.data import DataLoader
from utils.metrics import compute_accuracy, compute_f1, compute_f2
import argparse

# Reuse DummyCocoDataset for demonstration.
from scripts.train import DummyCocoDataset

def evaluate(model_path, batch_size=16):
    # Load your model – for demonstration we assume a state_dict load.
    model = CocoTransformerModule.load_from_checkpoint(model_path)
    model.eval()
    
    dataset = DummyCocoDataset(num_samples=100, num_classes=model.hparams.model.num_classes)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    all_acc, all_f1, all_f2 = [], [], []
    for batch in loader:
        x, y = batch
        logits = model(x)
        all_acc.append(compute_accuracy(y, logits))
        all_f1.append(compute_f1(y, logits))
        all_f2.append(compute_f2(y, logits))
    print("Accuracy:", sum(all_acc)/len(all_acc))
    print("F1 score:", sum(all_f1)/len(all_f1))
    print("F2 score:", sum(all_f2)/len(all_f2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint)
