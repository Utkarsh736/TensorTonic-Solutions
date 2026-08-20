from typing import Union

import numpy as np

def sigmoid(x: Union[float, list, np.ndarray]) -> np.ndarray:
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    return 1/(1+np.exp(-x))