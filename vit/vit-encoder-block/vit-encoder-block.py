import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    def layernorm(tensor, eps=1e-6):
        mean = np.mean(tensor, axis=-1, keepdims=True)
        var = np.var(tensor, axis=-1, keepdims=True)

        return (tensor-mean)/np.sqrt(var+eps)

    def gelu(tensor):
        return 0.5 * tensor * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (tensor + 0.044715 * np.power(tensor, 3))))

    B,N,D = x.shape
    head_dims = D//num_heads

    x_ln1 = layernorm(x)

    # Projection
    Q = np.matmul(x_ln1, Wq)
    K = np.matmul(x_ln1, Wk)
    V = np.matmul(x_ln1, Wv)

    # Reshape
    Q = Q.reshape(B, N, num_heads, head_dims).transpose(0,2,1,3)
    K = K.reshape(B, N, num_heads, head_dims).transpose(0,2,1,3)
    V = V.reshape(B, N, num_heads, head_dims).transpose(0,2,1,3)

    scores = np.matmul(Q, K.transpose(0,1,3,2))/np.sqrt(head_dims)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores-scores_max)
    attn_wghts = exp_scores/np.sum(exp_scores, axis=-1, keepdims=True)

    attn_out = np.matmul(attn_wghts, V)
    attn_out = attn_out.transpose(0,2,1,3).reshape(B,N,D)
    attn_out = np.matmul(attn_out, Wo)

    x_res1 = x + attn_out

    x_ln2 = layernorm(x_res1)

    mlp_out = gelu(np.matmul(x_ln2, W1))
    mlp_out = np.matmul(mlp_out, W2)

    out = x_res1 + mlp_out

    return out
    