import numpy as np

def tangent(coordinate, dirichlet, u_h):
    """
    Projects u_h to satisfy tangential conditions along Dirichlet edges.

    Parameters:
    - coordinate: (n_nodes, 2) array of node positions
    - dirichlet: (n_edges, 2) array of Dirichlet edges (node indices)
    - u_h: (n_nodes, 2) vector field at nodes (to be modified)

    Returns:
    - Modified u_h with tangential conditions imposed at boundary nodes
    """
    n_nodes = coordinate.shape[0]
    Tangent = np.zeros((2 * n_nodes, 2))
    eps_in_direction_edge_d = np.zeros((2 * n_nodes, 2))

    # Loop through each Dirichlet edge
    for j in range(dirichlet.shape[0]):
        n1, n2 = dirichlet[j]
        edge_vec = coordinate[n2] - coordinate[n1]
        edge_len = np.linalg.norm(edge_vec)
        tangent_vec = edge_vec * edge_len
        tan_eps = (1 / 1e6) * edge_vec / edge_len

        # Assign tangent and perturbation
        if np.linalg.norm(Tangent[2 * n1 - 1]) > 0:
            Tangent[2 * n1] = tangent_vec
            eps_in_direction_edge_d[2 * n1] = tan_eps
        else:
            Tangent[2 * n1 - 1] = tangent_vec
            eps_in_direction_edge_d[2 * n1 - 1] = tan_eps

        if np.linalg.norm(Tangent[2 * n2 - 1]) > 0:
            Tangent[2 * n2] = tangent_vec
            eps_in_direction_edge_d[2 * n2] = -tan_eps
        else:
            Tangent[2 * n2 - 1] = tangent_vec
            eps_in_direction_edge_d[2 * n2 - 1] = -tan_eps

    # Identify all unique Dirichlet nodes
    maske = np.zeros(n_nodes, dtype=int)
    maske[dirichlet.flatten()] = 1
    dirichletNode = np.where(maske == 1)[0]

    for j in dirichletNode:
        T = Tangent[2 * j - 1 : 2 * j + 1]  # two tangent rows
        if np.linalg.det(T) != 0:
            # Use tangents to reconstruct u_h
            v1 = coordinate[j] + eps_in_direction_edge_d[2 * j - 1]
            v2 = coordinate[j] + eps_in_direction_edge_d[2 * j]
            rhs = np.array([duds(v1), duds(v2)])
            u_h[j] = np.linalg.solve(T, rhs).T
        else:
            # Use tangent + normal for reconstruction
            t = Tangent[2 * j - 1]
            normal = t @ np.array([[0, 1], [-1, 0]])  # 90 deg rotation
            rhs = np.array([duds(coordinate[j]),
                            u_h[j] @ normal])
            u_h[j] = np.linalg.solve(np.vstack([t, normal]), rhs)

    return u_h
