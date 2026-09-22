import torch

def tanh_forward_backward(x, upstream_gradient):
    """
    Returns: tanh output and its upstream-scaled input gradient
    """
    y = torch.tanh(x)
    local = 1 - y.square()

    input_grad = upstream_gradient * local

    return (y, input_grad)
