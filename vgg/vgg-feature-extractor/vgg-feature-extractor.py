import numpy as np

def vgg_features(x: np.ndarray, config: list,
                 kernels: list, biases: list) -> np.ndarray:
    """
    Returns the float64 VGG features in NHWC layout.
    """
    curr_x = x
    k_idx = 0 # kernal ID

    for layer in config:
        if layer == 'M':
            N, H, W, C = curr_x.shape
            reshaped = curr_x.reshape(N, H//2, 2, W//2, 2, C)
            curr_x = reshaped.max(axis=(2, 4))

        else:
            W_k = kernels[k_idx]
            b_k = biases[k_idx]
            k_idx += 1

            N, H, W, in_C = curr_x.shape
            k_H, k_W, _, out_C = W_k.shape

            pad_H = (k_H - 1)//2
            pad_W = (k_W - 1)//2

            padded_x = np.pad(
                curr_x,
                ((0,0), (pad_H, pad_H), (pad_W, pad_W), (0, 0)),
                mode='constant',
            )

            conv_out = np.zeros((N, H, W, out_C), dtype=np.float64)
            for i in range(H):
                for j in range(W):
                    patch = padded_x[:, i:i+k_H, j:j+k_W, :]

                    conv_out[:, i, j, :] = np.sum(
                        patch[..., np.newaxis] * W_k[np.newaxis, ...],
                        axis=(1,2,3)
                    )

            curr_x = np.maximum(0.0, conv_out + b_k)

    return curr_x