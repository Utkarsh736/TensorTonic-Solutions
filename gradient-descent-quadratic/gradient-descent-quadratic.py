def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Return final x after 'steps' iterations.
    """

    curr_x = x0
    
    for i in range(steps):
        grad = 2*a*curr_x + b

        curr_x = curr_x - lr*grad
        

    return curr_x
        
