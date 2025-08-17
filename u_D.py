import numpy as np

def u_D(x):
    """
    Dirichlet boundary condition function (currently zero everywhere).

    Parameters:
    - x: (n, d) NumPy array of spatial points

    Returns:
    - dir: (n,) NumPy array of zeros
    """
    return np.zeros(x.shape[0])
