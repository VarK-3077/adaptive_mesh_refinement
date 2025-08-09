import numpy as np
from scipy.sparse import lil_matrix

def edge(element, coordinate):
    n_nodes = coordinate.shape[0]
    n_elements = element.shape[0]

    # Step 1: nodes2element sparse matrix
    nodes2element = lil_matrix((n_nodes, n_nodes), dtype=int)
    for j in range(n_elements):
        tri = element[j]
        for k in range(3):
            i1 = tri[k]
            i2 = tri[(k + 1) % 3]
            nodes2element[i1, i2] += (j + 1)  # MATLAB 1-based, adjust for Python indexing

    # Step 2: Symmetric matrix B
    B = nodes2element + nodes2element.T
    B = B.tocsr()

    # Step 3: Find upper triangle (i < j) non-zero entries → edges
    i, j = B.nonzero()
    mask = i < j
    i, j = i[mask], j[mask]

    noedges = len(i)

    # Step 4: Build nodes2edge
    nodes2edge = lil_matrix((n_nodes, n_nodes), dtype=int)
    for idx, (a, b) in enumerate(zip(i, j)):
        nodes2edge[a, b] = idx + 1
        nodes2edge[b, a] = idx + 1  # Symmetric

    # Step 5: edge2element matrix
    edge2element = np.zeros((noedges, 4), dtype=int)

    for m in range(n_elements):
        tri = element[m]
        for k in range(3):
            a = tri[k]
            b = tri[(k + 1) % 3]
            edge_id = nodes2edge[a, b] - 1  # Convert to 0-based
            if edge2element[edge_id, 0] == 0:
                edge2element[edge_id, :] = [
                    a, b,
                    nodes2element[a, b],
                    nodes2element[b, a]
                ]

    # Step 6: Interior and exterior edges
    interioredge = edge2element[edge2element[:, 3] > 0]
    exterioredge = edge2element[edge2element[:, 3] == 0][:, [0, 1, 2]]

    return nodes2element, nodes2edge, noedges, edge2element, interioredge
