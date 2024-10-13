import torch


def to_onehot(tensor: torch.Tensor) -> torch.Tensor:
    onehot = torch.zeros_like(tensor)
    onehot.scatter(1, tensor.unsqueeze(1), 1)

    return onehot
