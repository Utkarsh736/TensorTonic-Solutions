import torch

def neuron_backward(inputs, weights, bias, upstream_gradient):
    """
    Returns: output, input gradients, weight gradients, and bias gradient
    """
    a = torch.sum(weights*inputs) + bias
    y = torch.tanh(a)

    delta = upstream_gradient*(1-y.square())

    ip_grad = delta * weights
    op_grad = delta * inputs
    bias_grad = delta

    return(y, ip_grad, op_grad, bias_grad)
