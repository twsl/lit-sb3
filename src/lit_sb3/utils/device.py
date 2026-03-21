"""PyTorch device selection utilities."""

import torch


def get_device(device: torch.device | str = "auto") -> torch.device:
    """Retrieve PyTorch device.

    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    Args:
        device: One for 'auto', 'cuda', 'cpu'

    Returns:
        Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device
