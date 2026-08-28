import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224, crop_y: int = None, crop_x: int = None) -> np.ndarray:
    """
    Extract a crop from the image at (crop_y, crop_x). If not given, choose randomly.
    """
    H,W,C = image.shape

    max_x = W-crop_size
    max_y = H-crop_size

    if crop_x is None:
        crop_x = np.random.randint(0, max_x+1)
    
    if crop_y is None:
        crop_y = np.random.randint(0, max_y+1)

    return image[crop_y: crop_y+crop_size, crop_x: crop_x+crop_size, :]


def random_horizontal_flip(image: np.ndarray, p: float = 0.5, flip_rand: float = None) -> np.ndarray:
    """
    Flip image horizontally if flip_rand < p. If flip_rand not given, generate randomly.
    """
    
    if flip_rand is None:
        flip_rand = np.random.rand()

    if flip_rand<p:
        return np.fliplr(image)

    return image