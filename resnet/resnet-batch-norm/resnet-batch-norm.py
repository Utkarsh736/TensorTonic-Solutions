import numpy as np

def norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mean)/np.sqrt(var+eps)
    return gamma*x_norm + beta

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)
    identity = x.copy()
    
    if mode == "post":
        out = x@W1
        out = norm(out, gamma1, beta1)
        out = np.maximum(0, out)
        out = out@W2
        out = norm(out, gamma2, beta2)
        out = out+identity
        out = np.maximum(0, out)
        return {"output": [[round(float(v), 4) for v in row] for row in out], "mode": "post"}

    else:
        out = norm(x, gamma1, beta1)
        out = np.maximum(0, out)
        out = out@W1
        out = norm(out, gamma2, beta2)
        out = np.maximum(0, out)
        out = out@W2
        out = out+identity
        return {"output": [[round(float(v), 4) for v in row] for row in out], "mode": "pre"}