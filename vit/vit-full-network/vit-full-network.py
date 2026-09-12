import numpy as np

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray, b_head: np.ndarray = None) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    def layer_norm(values):
        mean = np.mean(values, axis=-1, keepdims=True)
        variance = np.mean((values - mean) ** 2, axis=-1, keepdims=True)
        return (values - mean) / np.sqrt(variance + 1e-6)

    def softmax(values):
        shifted = values - np.max(values, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    batch, height, width, channels = image.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    usable = image[:, :grid_height * patch_size, :grid_width * patch_size, :]
    patches = usable.reshape(batch, grid_height, patch_size, grid_width, patch_size, channels)
    patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(batch, grid_height * grid_width, -1)
    tokens = patches @ W_patch + patch_bias
    tokens = np.concatenate((np.broadcast_to(cls_token, (batch, 1, W_patch.shape[1])), tokens), axis=1)
    tokens = tokens + pos_embed

    for weights in encoder_weights:
        normalized = layer_norm(tokens)
        head_width = tokens.shape[-1] // num_heads
        q = (normalized @ weights["Wq"]).reshape(batch, -1, num_heads, head_width).transpose(0, 2, 1, 3)
        k = (normalized @ weights["Wk"]).reshape(batch, -1, num_heads, head_width).transpose(0, 2, 1, 3)
        v = (normalized @ weights["Wv"]).reshape(batch, -1, num_heads, head_width).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(head_width)
        attended = (softmax(scores) @ v).transpose(0, 2, 1, 3).reshape(batch, -1, tokens.shape[-1])
        tokens = tokens + attended @ weights["Wo"]
        normalized = layer_norm(tokens)
        hidden = normalized @ weights["W1"]
        gelu = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
        tokens = tokens + gelu @ weights["W2"]

    return layer_norm(tokens[:, 0, :]) @ W_head