import numpy as np

def f(x):
    """
    Volume force (source term) function.

    Parameters:
    - x: (n, d) NumPy array of spatial points

    Returns:
    - volumeforce: (n,) NumPy array of zeros
    """
    return np.zeros(x.shape[0])
