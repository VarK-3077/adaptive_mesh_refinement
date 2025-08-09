import numpy as np

def fluxEB(element, coordinate, u, noedges, nodes2edge, edge2element):
    """
    Compute the flux vector field from the solution vector u.

    Parameters:
    - element: (n_elements, 3) array of triangle node indices
    - coordinate: (n_nodes, 2) array of node coordinates
    - u: (noedges + n_elements,) array, solution vector (first noedges are flux dofs)
    - noedges: number of edges
    - nodes2edge: (n_nodes, n_nodes) symmetric matrix of edge indices (1-based in MATLAB)
    - edge2element: (noedges, 4) array of [n1, n2, elem1, elem2]

    Returns:
    - p: (3 * n_elements, 2) array of flux vectors on each triangle edge
    """
    n_elements = element.shape[0]
    p = np.zeros((3 * n_elements, 2))

    for j in range(n_elements):
        tri = element[j]
        coord = coordinate[tri].T  # shape (2, 3)

        # Local edges: [2→3, 3→1, 1→2]
        edge_ids = [nodes2edge[tri[1], tri[2]],
                    nodes2edge[tri[2], tri[0]],
                    nodes2edge[tri[0], tri[1]]]
        edge_ids = np.array(edge_ids) - 1  # zero-based

        signum = np.ones(3)
        for k in range(3):
            if edge2element[edge_ids[k], 3] == j + 1:  # MATLAB 1-based
                signum[k] = -1

        # Edge vectors
        c = coordinate[tri[[1, 2, 0]]] - coordinate[tri[[2, 0, 1]]]
        n = np.linalg.norm(c, axis=1)

        # Compute N matrix
        N = np.tile(coord.flatten()[:, np.newaxis], (1, 3)) - np.tile(coord, 3)

        # Edge-based DOFs
        u_e = u[edge_ids]

        # Scaling and assembling
        S = signum * n * u_e
        area_matrix = np.vstack([np.ones(3), coord])  # shape (3, 3)
        det_area = np.linalg.det(area_matrix)

        pc = (N @ np.diag(S)) / det_area
        pc = pc.reshape(2, 3)

        # Store in output
        p[3 * j : 3 * j + 3, :] = pc.T

    return p
