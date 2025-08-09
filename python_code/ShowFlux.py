import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ShowFlux(element, coordinate, p):
    """
    Visualize the x- and y-components of the approximate flux p over the mesh.

    Parameters:
    - element: (n_elements, 3) array of triangle vertex indices
    - coordinate: (n_nodes, 2) array of node coordinates
    - p: (3 * n_elements, 2) array of flux values per triangle (3 points per element)
    """
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')

    for j in range(element.shape[0]):
        tri_nodes = element[j]
        x = coordinate[tri_nodes, 0]
        y = coordinate[tri_nodes, 1]
        
        # Extract px and py values for triangle j
        px = p[3 * j : 3 * j + 3, 0]
        py = p[3 * j : 3 * j + 3, 1]

        # Plot px
        ax1.plot_trisurf(x, y, px, triangles=[[0, 1, 2]], color='lightblue', edgecolor='k', linewidth=0.3, alpha=0.9)
        ax2.plot_trisurf(x, y, py, triangles=[[0, 1, 2]], color='lightcoral', edgecolor='k', linewidth=0.3, alpha=0.9)

    ax1.set_title("p_x")
    ax2.set_title("p_y")

    for ax in [ax1, ax2]:
        ax.view_init(elev=50, azim=-60)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('flux')

    plt.tight_layout()
    plt.show()
