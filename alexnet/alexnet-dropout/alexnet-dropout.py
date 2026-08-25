import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, mask: np.ndarray = None) -> np.ndarray:
    """
    Apply inverted dropout. If mask is provided, use it; otherwise generate one.
    """
    if not training: return x

    if p>=1.0: return np.zeros_like(x)

    if mask is None:
        mask = np.random.binomial(1,1-p, size=x.shape)
    else:
        mask = np.asarray(mask)
        
    scale = 1.0/(1.0 - p)
    
    return x*mask*scale