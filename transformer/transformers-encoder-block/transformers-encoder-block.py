import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x-mean)/np.sqrt(var+eps)
    op = x_norm*gamma + beta
    
    return op

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model//num_heads

    # Projection
    q_proj = np.dot(Q, W_q)
    v_proj = np.dot(V, W_v)
    k_proj = np.dot(K, W_k)

    # Reshape
    q_heads = q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    v_heads = v_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    k_heads = k_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)

    scores = np.matmul(q_heads, k_heads.transpose(0,1,3,2))/np.sqrt(d_k)
    attn_wghts = softmax(scores, axis=-1)

    cntxt_hds = np.matmul(attn_wghts, v_heads)
    cntxt = cntxt_hds.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)

    op = np.dot(cntxt, W_o)

    return op

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.maximum(0, np.dot(x, W1)+b1)
    op = np.dot(hidden, W2)+b2

    return op

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    mha_out = multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    x = layer_norm(x+mha_out, gamma1, beta1)

    ffn_out = feed_forward(x,W1,b1,W2,b2)
    x = layer_norm(x+ffn_out, gamma2, beta2)

    return x