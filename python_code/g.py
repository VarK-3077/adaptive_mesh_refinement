import numpy as np

def g(x, n):
    """
    Neumann boundary condition function.

    Parameters:
    - x: (2,) NumPy array, midpoint of the edge [x, y]
    - n: (2,) NumPy array, outward normal vector at the edge

    Returns:
    - val: scalar, Neumann flux value at point x in direction n
    """
    # Convert x into complex form and rotate by (-1 - i)/sqrt(2)
    z = x[0] + 1j * x[1]
    z_rot = z * complex(-1, -1) / np.sqrt(2)

    # Compute angle and radius
    a = np.angle(z_rot) + 3 * np.pi / 4
    r = np.linalg.norm(x)

    # Compute directional value and project onto n
    val_vector = (2 / 3) * r ** (-1/3) * np.array([-np.sin(a / 3), np.cos(a / 3)])
    return np.dot(val_vector, n)
