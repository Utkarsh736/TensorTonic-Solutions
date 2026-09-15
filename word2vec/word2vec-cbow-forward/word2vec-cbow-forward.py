import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int,
                 W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 CBOW cross-entropy loss.
    """
    h = W_in[context_ids].mean(dim=0)
    logits = torch.matmul(W_out, h)
    log_prob = F.log_softmax(logits, dim=0)

    return -log_prob[target_id]