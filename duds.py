import numpy as np

def duds(x):
    """
    Returns a zero vector matching the number of input rows.
    
    Parameters:
    - x: (n, d) NumPy array (n points in d-dimensional space)

    Returns:
    - dir: (n,) NumPy array of zeros
    """
    return np.zeros(x.shape[0])
