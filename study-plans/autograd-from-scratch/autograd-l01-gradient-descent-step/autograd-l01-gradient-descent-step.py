import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)

    new_values = values - learning_rate*gradients
    update = new_values-values

    pred = np.sum(gradients*update)

    return ([float(x) for x in new_values], float(pred))
    
