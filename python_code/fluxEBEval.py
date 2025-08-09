import numpy as np

def fluxEBEval(element, coordinate, u, nodes2edge, edge2element):
    """
    Evaluate the approximate flux field at triangle quadrature points.

    Parameters:
    - element: (n_elements, 3) triangle node indices
    - coordinate: (n_nodes, 2) node coordinates
    - u: (noedges,) flux coefficients (edge DOFs)
    - nodes2edge: (n_nodes, n_nodes) edge ID lookup
    - edge2element: (noedges, 4) array of edge-node-element relationships

    Returns:
    - pEval: (3 * n_elements, 2) flux evaluated per element (3 values per triangle)
    """
    n_elements = element.shape[0]
    pEval = np.zeros((3 * n_elements, 2))

    for j in range(n_elements):
        tri = element[j]
        coords = coordinate[tri]  # (3, 2)
        
        # Compute signed area of the triangle
        area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))

        # Get edge indices: [2→3, 3→1, 1→2]
        e_ids = [nodes2edge[tri[1], tri[2]],
                 nodes2edge[tri[2], tri[0]],
                 nodes2edge[tri[0], tri[1]]]
        e_ids = np.array(e_ids) - 1  # convert to 0-based

        # Compute signum
        signum = np.ones(3)
        for k in range(3):
            if edge2element[e_ids[k], 3] == j + 1:  # MATLAB 1-based
                signum[k] = -1

        # Edge lengths
        edges = coords[[1, 2, 0]] - coords[[2, 0, 1]]
        edge_lengths = np.linalg.norm(edges, axis=1)  # [|e1|, |e2|, |e3|]

        # Mass-lumped quadrature matrix
        W = np.array([[4, 1, 1],
                      [1, 4, 1],
                      [1, 1, 4]]) / 6.0

        # Evaluate coordinate at quadrature point (3x2 matrix)
        coordEval = W @ coords  # shape (3, 2)

        # Flattened version and coordinate differences for N matrix
        coord1 = coords.T  # (2, 3)
        coordEval_flat = coordEval.T.flatten()[:, np.newaxis]  # (6, 1)
        N = coordEval_flat @ np.ones((1, 3)) - np.tile(coord1, 3)

        # Compute PCC and reshape
        factor = 1 / (2 * area)
        diag_sign_len_u = np.diag(signum * edge_lengths * u[e_ids])
        PCC = N @ (factor * diag_sign_len_u)  # (6, 1)
        PC = PCC.reshape(2, 3)

        # Store flux vector per element (3 rows for this triangle)
        pEval[3 * j : 3 * j + 3, :] = PC.T

    return pEval
