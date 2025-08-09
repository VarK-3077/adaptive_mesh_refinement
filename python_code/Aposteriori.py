import numpy as np

def Aposteriori(element, coordinate, dirichlet, Neumann, u, pEval):
    n_nodes = coordinate.shape[0]
    n_elements = element.shape[0]
    
    u_h = np.zeros((n_nodes, 2))
    supp_area = np.zeros(n_nodes)

    # Compute triangle centroids (WeightedPoint) – not used in this code
    WeightedPoint = np.mean(coordinate[element], axis=1)

    # Loop over all triangles
    for j in range(n_elements):
        nodes = element[j]
        coords = coordinate[nodes]
        area = 0.5 * np.linalg.det(np.array([[1, 1, 1], coords.T]))

        supp_area[nodes] += area / 3.0

        Pe = pEval[3 * j : 3 * j + 3, :]  # Local pEval (3x2)
        W = np.array([[4, 1, 1],
                      [1, 4, 1],
                      [1, 1, 4]]) / 36.0
        u_local = Pe.T @ W  # (2x3)
        u_h[nodes] += area * u_local.T

    # Normalize by support area
    temp_uh = u_h / supp_area[:, None]

    # Project to tangential component
    u_h = tangent(coordinate, dirichlet, temp_uh)

    # Error estimator per triangle
    eta_T = np.zeros(n_elements)
    for j in range(n_elements):
        nodes = element[j]
        coords = coordinate[nodes]
        area = 0.5 * np.linalg.det(np.array([[1, 1, 1], coords.T]))

        uh_local = u_h[nodes]           # (3x2)
        Pe = pEval[3 * j : 3 * j + 3]   # (3x2)

        W = np.array([[4, 1, 1],
                      [1, 4, 1],
                      [1, 1, 4]]) / 6.0

        diff = W @ uh_local - Pe        # (3x2)
        square_diff = np.sum(diff**2, axis=1)  # shape (3,)
        eta_T[j] = np.sqrt(area * np.sum(square_diff) / 6.0)

    # Global estimator
    eta_T = np.sqrt(np.sum(eta_T**2))
    
    return eta_T
