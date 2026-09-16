import numpy as np

def vgg_conv_block(x: np.ndarray, kernels: list,
                   biases: list) -> np.ndarray:
    """
    Returns the float64 block activations in NHWC layout.
    """
    op = x
    for kernel, bias in zip(kernels, biases):
        N, H, W, _ = op.shape
        kh, kw, _, c_out = kernel.shape

        pad_h = kh//2
        pad_w = kw//2
        padded = np.pad(op, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

        res = np.zeros((N,H, W, c_out))

        for i in range(H):
            for j in range(W):
                patch = padded[:, i:i+kh, j:j+kw, :]

                val = np.tensordot(patch, kernel, axes=((1,2,3), (0,1,2)))

                res[:, i, j, :] = val+bias

        op = np.maximum(res, 0.0)

    return op