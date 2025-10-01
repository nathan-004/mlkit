"""Generate random continuous datasets"""

import numpy as np

def linear_dataset(start:float = 0, end:float = 10, xstep:float = 1, a:float = 1, b:float = 1, ynoise:float = 3, seed:int = None):
    """
    Generate a random dataset around the linear function ax + b with random noise.
    Returns numpy arrays for x and y.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.arange(start, end, xstep)
    noise = np.random.uniform(-ynoise, ynoise, size=x.shape)
    y = a * x + b + noise
    return x, y