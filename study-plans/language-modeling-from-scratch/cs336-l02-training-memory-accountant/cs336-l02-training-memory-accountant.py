import math

def total_elements(shapes):
  return sum([math.prod(shape) for shape in shapes])

def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:
    """
    Returns integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """
    param_n = total_elements(param_shapes)
    act_n = total_elements(activation_shapes)
      
    param = param_n * param_bytes_per_element
    grad = param_n * grad_bytes_per_element
    act = act_n * activation_bytes_per_element
    
    if optimizer == "sgd": buffer = 0
    elif optimizer == "adagrad": buffer = 1
    elif optimizer == "adam": buffer = 2
    
    opt_state = buffer * param_n * optimizer_bytes_per_element
    total = param + grad + act + opt_state
    
    return {"parameters": param, "gradients": grad, "activations": act,
          "optimizer_state": opt_state, "total": total}
