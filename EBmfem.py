import numpy as np
from scipy.sparse import lil_matrix, bmat
from scipy.sparse.linalg import spsolve

# Import custom FEM components
from edge import edge
from stimaB import stimaB
from f import f
from g import g
from u_D import u_D
from fluxEB import fluxEB
from fluxEBEval import fluxEBEval
from Aposteriori import Aposteriori
from tangent import tangent
from ShowDisplacement import ShowDisplacement
from ShowFlux import ShowFlux

def main():
    # --- Load mesh data ---
    coordinate = np.loadtxt("coordinate.dat")
    element = np.loadtxt("element.dat", dtype=int) - 1
    dirichlet = np.loadtxt("dirichlet.dat", dtype=int) - 1
    try:
        Neumann = np.loadtxt("Neumann.dat", dtype=int) - 1
        if Neumann.ndim == 1:
            Neumann = Neumann.reshape(1, -1)
    except OSError:
        Neumann = np.empty((0, 2), dtype=int)

    # --- Preprocessing ---
    nodes2element, nodes2edge, noedges, edge2element, interioredge = edge(element, coordinate)
    n_elements = element.shape[0]

    # --- Assemble B and C ---
    B = lil_matrix((noedges, noedges))
    C = lil_matrix((noedges, n_elements))

    for j in range(n_elements):
        elem_nodes = element[j]
        coord = coordinate[elem_nodes].T

    # Safely extract edge indices
        I = np.array([
            nodes2edge[elem_nodes[1], elem_nodes[2]],
            nodes2edge[elem_nodes[2], elem_nodes[0]],
            nodes2edge[elem_nodes[0], elem_nodes[1]]
        ]) - 1

    # Sign correction
        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1:
                signum[k] = -1

        B[np.ix_(I, I)] += np.diag(signum) @ stimaB(coord) @ np.diag(signum)
        edge_vecs = coord[:, [2, 0, 1]] - coord[:, [1, 2, 0]]
        C[I, j] = signum * np.linalg.norm(edge_vecs, axis=0)


    # --- Global matrix A ---
    A = bmat([[B, C], [C.T, None]], format='csr')

    # --- Right-hand side vector b ---
    b = np.zeros(noedges + n_elements)
    for j in range(n_elements):
        area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coordinate[element[j]].T]))
        centroid = np.mean(coordinate[element[j]], axis=0)
        b[noedges + j] = -area * f(centroid) / 3.0

    # --- Apply Dirichlet BCs ---
    for k in range(dirichlet.shape[0]):
        n1, n2 = dirichlet[k]
        edge_id = nodes2edge[n1, n2] - 1
        midpoint = (coordinate[n1] + coordinate[n2]) / 2
        length = np.linalg.norm(coordinate[n1] - coordinate[n2])
        b[edge_id] = length * u_D(midpoint)

    # --- Apply Neumann BCs ---
    if Neumann.size > 0:
        tmp = np.zeros(noedges + n_elements)
        neumann_edges = np.diag(nodes2edge[Neumann[:, 0], Neumann[:, 1]]) - 1
        tmp[neumann_edges] = 1
        FreeEdge = np.where(tmp == 0)[0]
        x = np.zeros(noedges + n_elements)

        CN = coordinate[Neumann[:, 1]] - coordinate[Neumann[:, 0]]
        normals = np.einsum('ij, jk -> ik', CN, np.array([[0, -1], [1, 0]]))
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]

        for j in range(Neumann.shape[0]):
            eid = nodes2edge[Neumann[j, 0], Neumann[j, 1]] - 1
            midpoint = np.mean(coordinate[Neumann[j]], axis=0)
            x[eid] = g(midpoint, normals[j])

        b -= A @ x
        x[FreeEdge] = spsolve(A[FreeEdge][:, FreeEdge], b[FreeEdge])
    else:
        x = spsolve(A, b)

    # --- Post-processing ---
    u = x[noedges:]
    ShowDisplacement(element, coordinate, u)
    p = fluxEB(element, coordinate, x[:noedges], noedges, nodes2edge, edge2element)
    ShowFlux(element, coordinate, p)
    pEval = fluxEBEval(element, coordinate, x[:noedges], nodes2edge, edge2element)
    eta_T = Aposteriori(element, coordinate, dirichlet, Neumann, x, pEval)
    print(f"A posteriori error estimator eta_T = {eta_T:.6e}")

if __name__ == "__main__":
    main()
