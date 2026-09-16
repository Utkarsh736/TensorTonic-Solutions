import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray,
                   W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    B = features.shape[0]
    flat_feat = features.reshape(B, -1)

    # Layer1: Linear+ReLU
    h1 = np.dot(flat_feat, W1) + b1
    a1 = np.maximum(0.0, h1)

    # Layer2: Linear + ReLU
    h2 = np.dot(a1, W2) + b2
    a2 = np.maximum(0.0, h2)

    # Layer3: Linear only
    logits = np.dot(a2, W3) + b3

    return logits

    