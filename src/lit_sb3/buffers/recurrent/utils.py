from collections.abc import Callable
from functools import partial

import numpy as np
import torch


def pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Chunk sequences and pad them to have constant dimensions.

    Args:
        seq_start_indices: Indices of the transitions that start a sequence
        seq_end_indices: Indices of the transitions that end a sequence
        device: PyTorch device
        tensor: Tensor of shape (batch_size, *tensor_shape)
        padding_value: Value used to pad sequence to the same length
            (zero padding by default)

    Returns:
        (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [
        torch.tensor(tensor[start : end + 1], device=device)
        for start, end in zip(seq_start_indices, seq_end_indices, strict=False)
    ]
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)


def pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Pad and flatten the sequences of scalar values,.

    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,).

    Args:
        seq_start_indices: Indices of the transitions that start a sequence
        seq_end_indices: Indices of the transitions that end a sequence
        device: PyTorch device (cpu, gpu, ...)
        tensor: Tensor of shape (max_length, n_seq, 1)
        padding_value: Value used to pad sequence to the same length
            (zero padding by default)

    Returns:
        (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()


def create_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, Callable, Callable]:
    """Create the utility function to chunk data into.

    sequences and pad them to create fixed size tensors.

    Args:
        episode_starts: Indices where an episode starts
        env_change: Indices where the data collected
            come from a different env (when using multiple env for data collection)
        device: PyTorch device

    Returns:
        Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])

    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten
