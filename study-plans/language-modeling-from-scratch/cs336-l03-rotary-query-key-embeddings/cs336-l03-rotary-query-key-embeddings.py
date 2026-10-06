import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """

    B, H, S, D = q.shape
    q_pairs = q.reshape(B, H, S, D // 2, 2)
    k_pairs = k.reshape(B, H, S, D // 2, 2)

    if positions.dim() == 1:      
        theta = positions[:, None] * inv_freq            # (S, D/2)
        theta = theta[None, None, :, :, None]            # (1, 1, S, D/2, 1)
    else:                          
        theta = positions[:, :, None] * inv_freq         # (B, S, D/2)
        theta = theta[:, None, :, :, None] 

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # Rotation
    x0 = q_pairs[..., 0:1]
    x1 = q_pairs[..., 1:2]
    
    x0_new = x0 * cos - x1 * sin
    x1_new = x0 * sin + x1 * cos

    q_rotated = torch.cat([x0_new, x1_new], dim=-1)   # (B, H, S, D/2, 2)
    q_rotated = q_rotated.flatten(-2)                 
    q_rotated = q_rotated.to(q.dtype)

    y0 = k_pairs[..., 0:1]
    y1 = k_pairs[..., 1:2]

    y0_new = y0 * cos - y1 * sin
    y1_new = y0 * sin + y1 * cos

    k_rotated = torch.cat([y0_new, y1_new], dim=-1)
    k_rotated = k_rotated.flatten(-2)
    k_rotated = k_rotated.to(k.dtype)

    return {"q_rotated": q_rotated, "k_rotated": k_rotated}