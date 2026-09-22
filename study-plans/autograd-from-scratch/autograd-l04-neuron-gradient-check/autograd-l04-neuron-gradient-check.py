import torch

def neuron_gradient_check(inputs, weights, bias, h):
    """
    Returns: analytic and numerical parameter gradients with their maximum error
    """
    inputs = inputs.to(torch.float64)
    weights = weights.to(torch.float64)
    bias = bias.to(torch.float64)

    with torch.no_grad():
        def forward(inputs, weights, bias):
            a = torch.sum(weights*inputs) + bias
            return torch.tanh(a)
    
        y = forward(inputs, weights, bias)

        # Analytic
        an_weight_grads = (1 - y.square())*inputs
        an_bias_grad = 1 - y.square()

        # Numerical
        num_weight_grads = torch.zeros_like(weights)
        for i in range(len(weights)):
            w_p = weights.clone()
            w_p[i] = w_p[i]+h
    
            y_p = forward(inputs, w_p, bias)
            
            num_weight_grads[i] = (y_p - y)/h
    
        # For bias
        y_p = forward(inputs, weights, bias+h)
        num_bias_grad = (y_p - y)/h
    
        # max error
        diff_w = (an_weight_grads - num_weight_grads).abs()
        max_err_w = diff_w.max() if diff_w.numel()>0 else torch.tensor(0.0, dtype=torch.float64)
        max_err_b = (an_bias_grad - num_bias_grad).abs()
        max_err = torch.maximum(max_err_w, max_err_b)
    
        return (an_weight_grads, num_weight_grads, an_bias_grad, num_bias_grad, max_err)
