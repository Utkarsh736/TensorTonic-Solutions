import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    batch_size = image.shape[0]

    pddng = 2
    krnl_sz = 11
    stride = 4
    filters=96

    op_ht = (image.shape[1] + 2*pddng -krnl_sz)//stride+1
    op_wdth = (image.shape[2] + 2*pddng - krnl_sz)//stride+1
    
    op_shp = (batch_size, op_ht, op_wdth, filters)
    return np.zeros(op_shp)