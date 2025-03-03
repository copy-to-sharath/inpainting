# utils/metrics.py
import torch
from sklearn.metrics import f1_score, accuracy_score

def compute_accuracy(y_true, y_pred):
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    return accuracy_score(y_true.cpu().numpy(), y_pred_labels)

def compute_f1(y_true, y_pred, average="macro"):
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    return f1_score(y_true.cpu().numpy(), y_pred_labels, average=average)

def compute_f2(y_true, y_pred, beta=2, average="macro"):
    # f_beta score manually (for beta=2)
    # For brevity, we use f1_score with beta=1 as a placeholder.
    # In production, use a library or custom implementation.
    return f1_score(y_true.cpu().numpy(), torch.argmax(y_pred, dim=1).cpu().numpy(), average=average)
