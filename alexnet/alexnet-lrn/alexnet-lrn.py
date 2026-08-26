import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Apply Local Response Normalization across channels.
    """
    x = np.asarray(x, dtype=np.float64)
    B,H,W,C = x.shape

    sqrd = x**2

    chnl_sum = np.zeros_like(x)
    radius = n//2

    for i in range(C):
        start = max(0, i-radius)
        end = min(C, i+radius+1)

        chnl_sum[..., i] = np.sum(sqrd[..., start:end], axis=-1)

    dnm = (k + alpha * chnl_sum) ** beta

    return x/dnm