import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    B,H,W,C = image.shape

    H_new = (H//patch_size) * patch_size
    W_new = (W//patch_size) * patch_size

    image = image[:, :H_new, :W_new, :]

    num_h = H_new//patch_size
    num_w = W_new//patch_size

    reshaped = image.reshape(B, num_h, patch_size, num_w, patch_size, C)
    transposed = reshaped.transpose(0,1,3,2,4,5)

    # Flatten
    N = num_h*num_w
    flattened = transposed.reshape(B, N, patch_size*patch_size*C)

    out = np.matmul(flattened, W_proj) + bias

    return out