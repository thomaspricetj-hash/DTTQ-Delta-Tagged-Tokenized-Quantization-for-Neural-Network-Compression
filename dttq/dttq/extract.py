import torch
from typing import List, Tuple

def extract_matrices(model) -> List[Tuple[str, torch.Tensor]]:
    matrices = []
    for name, param in model.named_parameters():
        if len(param.shape) == 2:
            matrices.append((name, param.detach().cpu()))
    return matrices
