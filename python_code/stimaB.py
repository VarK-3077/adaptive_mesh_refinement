import numpy as np
from scipy.sparse import diags

def stimaB(coord):
    coord = np.array(coord)  # Ensure it's a NumPy array (shape should be 3x2)
    
    # Step 1: Create N = coord(:) * ones(1,3) - repmat(coord, 3, 1)
    # In Python: flatten and reshape as needed
    N = np.tile(coord.flatten()[:, np.newaxis], (1, 3)) - np.tile(coord.T, 3)

    # Step 2: Diagonal matrix D from specific norms
    d1 = np.linalg.norm(N[[4, 5], 1])  # MATLAB's N([5,6],2) is Python N[[4,5], 1]
    d2 = np.linalg.norm(N[[0, 1], 2])  # N([1,2],3) → N[[0,1],2]
    d3 = np.linalg.norm(N[[0, 1], 1])  # N([1,2],2) → N[[0,1],1]
    D = np.diag([d1, d2, d3])

    # Step 3: Create sparse matrix M
    diags_data = [
        np.ones(6),              # -4
        np.ones(6),              # -2
        2 * np.ones(6),          #  0
        np.ones(6),              #  2
        np.ones(6)               #  4
    ]
    M = diags(diags_data, offsets=[-4, -2, 0, 2, 4], shape=(6, 6)).toarray()

    # Step 4: Compute determinant of area matrix
    area_matrix = np.vstack([np.ones(3), coord.T])  # [1,1,1;coord]
    detA = np.linalg.det(area_matrix)

    # Step 5: Compute B
    B = D @ N.T @ M @ N @ D / (24 * detA)

    return B
