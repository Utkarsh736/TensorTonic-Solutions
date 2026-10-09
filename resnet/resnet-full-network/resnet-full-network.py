import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.array(x, dtype=float)
    conv1 = np.array(conv1, dtype=float)
    fc = np.array(fc, dtype=float)
    W1_b1 = np.array(W1_b1, dtype=np.float64)
    W2_b1 = np.array(W2_b1, dtype=np.float64)
    W1_b2 = np.array(W1_b2, dtype=np.float64)
    W2_b2 = np.array(W2_b2, dtype=np.float64)
    Ws_b2 = np.array(Ws_b2, dtype=np.float64)
    
    h = ReLU(x @ conv1)
    shortcut = h
    out1 = ReLU(ReLU(h @ W1_b1) @ W2_b1 + shortcut)
    shortcut = out1 @ Ws_b2
    out2 = ReLU(ReLU(out1 @ W1_b2) @ W2_b2 + shortcut)

    logits = out2 @ fc
    return [[round(float(v), 4) for v in row] for row in logits]