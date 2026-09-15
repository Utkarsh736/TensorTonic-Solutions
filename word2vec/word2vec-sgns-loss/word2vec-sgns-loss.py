import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 SGNS loss.
    """
    pos = torch.dot(center_vec, pos_vec)
    neg = torch.mv(neg_vecs, center_vec)

    # F.softplus(-x) = -log(sigmoid(x))
    p_loss = F.softplus(-pos)    
    n_loss = F.softplus(neg)

    return p_loss + torch.sum(n_loss)