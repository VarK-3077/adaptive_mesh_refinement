import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ShowDisplacement(element, coordinate, u):
    """
    Visualize the approximate displacement u on the mesh.

    Parameters:
    - element: (n_elements, 3) array of triangle vertex indices
    - coordinate: (n_nodes, 2) array of node coordinates
    - u: (n_elements,) array of scalar displacement per element
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Loop over triangles and plot each with constant displacement
    for j in range(element.shape[0]):
        nodes = element[j]
        x = coordinate[nodes, 0]
        y = coordinate[nodes, 1]
        z = np.full(3, u[j])  # constant value over the triangle

        # Triangular surface
        ax.plot_trisurf(x, y, z, triangles=[[0, 1, 2]], color='lightblue', edgecolor='k', linewidth=0.5, alpha=0.8)

    ax.view_init(elev=50, azim=-60)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Displacement')
    plt.title('Approximate Displacement u')
    plt.tight_layout()
    plt.show()
