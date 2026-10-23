"""PyTorch device selection utilities."""

import torch


def get_device(device: torch.device | str = "auto") -> torch.device:
    """Retrieve PyTorch device.

    It checks that the requested device is available first.
    By default, it prefers CUDA, then MPS, then CPU.

    Args:
        device: One of 'auto', 'cuda', 'mps', or 'cpu'

    Returns:
        Supported Pytorch device
    """
    # Prefer CUDA when available, then Apple MPS.
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    # Force conversion to torch.device
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")

    return device
