import torch

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    C.copy_(A + B)
