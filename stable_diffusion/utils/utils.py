# utils/utils.py
import torch

def save_model_pytorch_mar(model, file_path="model.mar", example_input=None):
    """
    Save model using TorchScript (MAR format).
    """
    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(file_path)

def save_model_huggingface(model, file_path="hf_model"):
    """
    Save model in a Hugging Face style. This usually requires that your model has a .config.
    For demonstration, we assume the model is a transformers.PreTrainedModel.
    """
    try:
        model.save_pretrained(file_path)
    except AttributeError:
        # If not available, simply save state dict.
        torch.save(model.state_dict(), file_path + ".bin")
