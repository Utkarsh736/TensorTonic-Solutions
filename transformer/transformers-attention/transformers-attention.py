import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = K.shape[-1]

    scores = torch.matmul(Q, K.transpose(-2, -1))

    scaled_scores = scores/math.sqrt(d_k)

    attn_wghts = F.softmax(scaled_scores, dim=-1)

    op = torch.matmul(attn_wghts, V)

    return op