
import numpy as np
from scipy.sparse import lil_matrix, bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Utility functions ---

def f(x):
    """Volume force function. Supports single point or batch of points."""
    if x.ndim == 1:
        return 0.0
    return np.zeros(x.shape[0])


def u_D(x):
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        return 0.0
    return np.zeros(x.shape[0])


def g(x, n):
    z = x[0] + 1j * x[1]
    z_rot = z * complex(-1, -1) / np.sqrt(2)
    a = np.angle(z_rot) + 3 * np.pi / 4
    r = np.linalg.norm(x)
    val_vector = (2 / 3) * r ** (-1/3) * np.array([-np.sin(a / 3), np.cos(a / 3)])
    return np.dot(val_vector, n)

def duds(x):
    """Dummy directional derivative for Dirichlet tangent condition"""
    x = np.atleast_1d(x)
    return 0.0


def stimaB(coord):
    # coord shape is (2, 3) = 2D coordinates of the triangle's 3 vertices
    v1 = coord[:, 0]
    v2 = coord[:, 1]
    v3 = coord[:, 2]

    N = np.zeros((6, 3))
    N[0:2, :] = np.tile(v1[:, np.newaxis], (1, 3)) - coord
    N[2:4, :] = np.tile(v2[:, np.newaxis], (1, 3)) - coord
    N[4:6, :] = np.tile(v3[:, np.newaxis], (1, 3)) - coord

    d1 = np.linalg.norm(N[4:6, 1])
    d2 = np.linalg.norm(N[0:2, 2])
    d3 = np.linalg.norm(N[0:2, 1])
    D = np.diag([d1, d2, d3])

    M = np.zeros((6, 6))
    for i in range(3):
        M[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * 2
    M[0:2, 2:4] = M[2:4, 0:2] = np.eye(2)
    M[2:4, 4:6] = M[4:6, 2:4] = np.eye(2)
    M[0:2, 4:6] = M[4:6, 0:2] = np.eye(2)

    area = np.linalg.det(np.vstack([np.ones(3), coord]))  # 3x3 matrix
    return D @ N.T @ M @ N @ D / (24 * area)


def edge(element, coordinate):
    n_nodes = coordinate.shape[0]
    n_elements = element.shape[0]
    nodes2element = lil_matrix((n_nodes, n_nodes), dtype=int)
    for j in range(n_elements):
        tri = element[j]
        for k in range(3):
            i1 = tri[k]
            i2 = tri[(k + 1) % 3]
            nodes2element[i1, i2] += (j + 1)
    B = nodes2element + nodes2element.T
    B = B.tocsr()
    i, j = B.nonzero()
    mask = i < j
    i, j = i[mask], j[mask]
    noedges = len(i)
    nodes2edge = lil_matrix((n_nodes, n_nodes), dtype=int)
    for idx, (a, b) in enumerate(zip(i, j)):
        nodes2edge[a, b] = idx + 1
        nodes2edge[b, a] = idx + 1
    edge2element = np.zeros((noedges, 4), dtype=int)
    for m in range(n_elements):
        tri = element[m]
        for k in range(3):
            a = tri[k]
            b = tri[(k + 1) % 3]
            edge_id = nodes2edge[a, b] - 1
            if edge2element[edge_id, 0] == 0:
                edge2element[edge_id, :] = [a, b, nodes2element[a, b], nodes2element[b, a]]
    interioredge = edge2element[edge2element[:, 3] > 0]
    return nodes2element, nodes2edge, noedges, edge2element, interioredge

def tangent(coordinate, dirichlet, u_h):
    n_nodes = coordinate.shape[0]
    Tangent = np.zeros((2 * n_nodes, 2))
    eps_dir = np.zeros((2 * n_nodes, 2))
    for edge in dirichlet:
        p1, p2 = edge
        v = coordinate[p2] - coordinate[p1]
        l = np.linalg.norm(v)
        tangent_vec = v * l
        eps = (1e-6) * v / l
        idx1 = 2 * p1
        idx2 = 2 * p2
        if np.linalg.norm(Tangent[idx1]) > 0:
            Tangent[idx1 + 1] = tangent_vec
            eps_dir[idx1 + 1] = eps
        else:
            Tangent[idx1] = tangent_vec
            eps_dir[idx1] = eps
        if np.linalg.norm(Tangent[idx2]) > 0:
            Tangent[idx2 + 1] = tangent_vec
            eps_dir[idx2 + 1] = -eps
        else:
            Tangent[idx2] = tangent_vec
            eps_dir[idx2] = -eps
    mask = np.zeros(n_nodes)
    mask[dirichlet.flatten()] = 1
    nodes = np.where(mask == 1)[0]
    for j in nodes:
        T = Tangent[2*j:2*j+2]
        if np.linalg.det(T) != 0:
            rhs = np.array([
                duds(coordinate[j] + eps_dir[2*j]),
                duds(coordinate[j] + eps_dir[2*j+1])
            ])
            u_h[j] = np.linalg.solve(T, rhs).T
        else:
            t = Tangent[2*j]
            n = t @ np.array([[0, 1], [-1, 0]])
            rhs = np.array([float(duds(coordinate[j])), np.dot(u_h[j], n)])
            u_h[j] = np.linalg.solve(np.vstack([t, n]), rhs)
    return u_h

def ShowDisplacement(element, coordinate, u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(element.shape[0]):
        tri = element[j]
        x = coordinate[tri, 0]
        y = coordinate[tri, 1]
        z = np.full(3, u[j])
        ax.plot_trisurf(x, y, z, triangles=[[0,1,2]], color='lightblue', edgecolor='k')
    ax.view_init(50, -60)
    plt.title('Displacement u')
    plt.show()

# Placeholder for fluxEB, fluxEBEval, Aposteriori
# (You can add them here similarly if desired.)



def fluxEB(element, coordinate, u, noedges, nodes2edge, edge2element):
    p = np.zeros((3 * element.shape[0], 2))
    for j in range(element.shape[0]):
        tri = element[j]
        coord = coordinate[tri].T  # shape (2,3)

        I = np.array([
            nodes2edge[tri[1], tri[2]],
            nodes2edge[tri[2], tri[0]],
            nodes2edge[tri[0], tri[1]]
        ]) - 1

        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1:
                signum[k] = -1

        c = coordinate[tri[[1, 2, 0]]] - coordinate[tri[[2, 0, 1]]]  # edge vectors
        n = np.linalg.norm(c, axis=1)  # edge lengths

        N = np.zeros((6, 3))
        for i in range(3):
            vi = coord[:, i].reshape(2, 1)
            N[2*i:2*i+2, :] = vi - coord

        area = np.linalg.det(np.vstack([np.ones(3), coord])) / 2
        PC = N @ np.diag(signum * n) @ u[I] / (2 * area)
        PC = PC.reshape(2, 3).T
        p[3 * j : 3 * j + 3, :] = PC
    return p


def fluxEBEval(element, coordinate, u, nodes2edge, edge2element):
    pEval = np.zeros((3 * element.shape[0], 2))
    for j in range(element.shape[0]):
        tri = element[j]
        coord = coordinate[tri].T  # shape (2, 3)

        area = np.linalg.det(np.vstack([np.ones(3), coord])) / 2

        I = np.array([
            nodes2edge[tri[1], tri[2]],
            nodes2edge[tri[2], tri[0]],
            nodes2edge[tri[0], tri[1]]
        ]) - 1

        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1:
                signum[k] = -1

        edge_vectors = coordinate[tri[[1, 2, 0]]] - coordinate[tri[[2, 0, 1]]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)

        # Midpoint evaluation
        weights = np.array([[4, 1, 1],
                            [1, 4, 1],
                            [1, 1, 4]]) / 6
        coordEval = coordinate[tri].T @ weights.T  # shape (2, 3)

        N = np.zeros((6, 3))
        for i in range(3):
            vi = coordEval[:, i].reshape(2, 1)
            N[2*i:2*i+2, :] = vi - coord

        PC = N @ (np.diag(signum * edge_lengths) @ u[I]) / (2 * area)
        PC = PC.reshape(2, 3).T
        pEval[3 * j : 3 * j + 3, :] = PC
    return pEval


def Aposteriori(element, coordinate, dirichlet, Neumann, u, pEval):
    n_nodes = coordinate.shape[0]
    n_elements = element.shape[0]
    u_h = np.zeros((n_nodes, 2))
    supp_area = np.zeros(n_nodes)
    for j in range(n_elements):
        nodes = element[j]
        coords = coordinate[nodes]
        area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))
        supp_area[nodes] += area / 3.0
        Pe = pEval[3 * j : 3 * j + 3, :]
        W = np.array([[4, 1, 1],
                      [1, 4, 1],
                      [1, 1, 4]]) / 36.0
        u_local = Pe.T @ W
        u_h[nodes] += area * u_local.T
    temp_uh = u_h / supp_area[:, None]
    u_h = tangent(coordinate, dirichlet, temp_uh)
    eta_T = np.zeros(n_elements)
    for j in range(n_elements):
        nodes = element[j]
        coords = coordinate[nodes]
        area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))

        uh_local = u_h[nodes]
        Pe = pEval[3 * j : 3 * j + 3]
        W = np.array([[4, 1, 1],
                      [1, 4, 1],
                      [1, 1, 4]]) / 6.0
        diff = W @ uh_local - Pe
        square_diff = np.sum(diff**2, axis=1)
        eta_T[j] = np.sqrt(area * np.sum(square_diff) / 6.0)
    eta_T = np.sqrt(np.sum(eta_T**2))
    return eta_T

def ShowFlux(element, coordinate, p):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')

    for j in range(element.shape[0]):
        tri_nodes = element[j]
        x = coordinate[tri_nodes, 0]
        y = coordinate[tri_nodes, 1]

        px = p[3 * j : 3 * j + 3, 0]
        py = p[3 * j : 3 * j + 3, 1]

        ax1.plot_trisurf(x, y, px, triangles=[[0, 1, 2]], color='lightblue', edgecolor='k', linewidth=0.3, alpha=0.9)
        ax2.plot_trisurf(x, y, py, triangles=[[0, 1, 2]], color='lightcoral', edgecolor='k', linewidth=0.3, alpha=0.9)

    ax1.set_title("Flux component p_x")
    ax2.set_title("Flux component p_y")

    for ax in [ax1, ax2]:
        ax.view_init(elev=50, azim=-60)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('flux')

    plt.tight_layout()
    plt.show()


def plot_mesh(coordinate, element, title="Mesh"):
    plt.figure(figsize=(6, 6))
    for tri in element:
        pts = coordinate[tri]
        pts = np.vstack([pts, pts[0]])  # close the triangle
        plt.plot(pts[:, 0], pts[:, 1], 'k-')
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

# -- Mark Elements for Refinement --
def mark_elements(eta, theta=0.5):
    sorted_idx = np.argsort(-eta)
    total = np.sum(eta)
    cumulative = np.cumsum(eta[sorted_idx])
    marked = sorted_idx[cumulative <= theta * total]
    if marked.size == 0:
        marked = sorted_idx[:1]
    return marked

# -- Refine Mesh --
def refine_mesh(coordinate, element, marked):
    new_coords = coordinate.tolist()
    midpoint_cache = {}
    new_elements = []
    for i, tri in enumerate(element):
        if i not in marked:
            new_elements.append(tri)
            continue
        midpoints = []
        mid_idx = []
        for j in range(3):
            a, b = sorted((tri[j], tri[(j + 1) % 3]))
            if (a, b) not in midpoint_cache:
                midpoint = 0.5 * (coordinate[a] + coordinate[b])
                midpoint_cache[(a, b)] = len(new_coords)
                new_coords.append(midpoint.tolist())
            mid_idx.append(midpoint_cache[(a, b)])
        a, b, c = tri
        ab, bc, ca = mid_idx
        new_elements.extend([
            [a, ab, ca],
            [ab, b, bc],
            [ca, bc, c],
            [ab, bc, ca]
        ])
    return np.array(new_coords), np.array(new_elements, dtype=int)

# -- Adaptive Solver --
def adaptive_RT(coordinate, element, dirichlet, Neumann, tol=1e-3, max_cycles=10):
    for cycle in range(max_cycles):
        print(f"\nAdaptive cycle {cycle+1}")
        plot_mesh(coordinate, element, title=f"Mesh at cycle {cycle+1}")
        
        nodes2element, nodes2edge, noedges, edge2element, interioredge = edge(element, coordinate)

        B = lil_matrix((noedges, noedges))
        C = lil_matrix((noedges, element.shape[0]))

        for j in range(element.shape[0]):
            tri = element[j]
            coord = coordinate[tri].T
            I = np.array([
                nodes2edge[tri[1], tri[2]],
                nodes2edge[tri[2], tri[0]],
                nodes2edge[tri[0], tri[1]]
            ]) - 1
            signum = np.ones(3)
            for k in range(3):
                if edge2element[I[k], 3] == j + 1:
                    signum[k] = -1
            B[np.ix_(I, I)] += np.diag(signum) @ stimaB(coord) @ np.diag(signum)
            n = coord[:, [2, 0, 1]] - coord[:, [1, 2, 0]]
            C[I, j] = signum * np.linalg.norm(n, axis=0)

        A = bmat([[B, C], [C.T, None]]).tocsc()
        b = np.zeros(noedges + element.shape[0])
        for j in range(element.shape[0]):
            coords = coordinate[element[j]]
            area = np.linalg.det(np.vstack([np.ones(3), coords.T]))
            b[noedges + j] = -area * f(np.mean(coords, axis=0)) / 6

        for k in range(dirichlet.shape[0]):
            i, j = dirichlet[k]
            idx = nodes2edge[i, j] - 1
            edge_len = np.linalg.norm(coordinate[i] - coordinate[j])
            b[idx] = edge_len * u_D((coordinate[i] + coordinate[j]) / 2)

        if Neumann.size > 0:
            tmp = np.zeros_like(b)
            for j in range(Neumann.shape[0]):
                i, k = Neumann[j]
                idx = nodes2edge[i, k] - 1
                normal = coordinate[k] - coordinate[i]
                rotated = normal @ np.array([[0, -1], [1, 0]]) / np.linalg.norm(normal)
                tmp[idx] = g((coordinate[i] + coordinate[k]) / 2, rotated)
            b -= A @ tmp
            free = np.where(tmp == 0)[0]
            x = tmp.copy()
            x[free] = spsolve(A[free, :][:, free], b[free])
        else:
            x = spsolve(A, b)

        pEval = fluxEBEval(element, coordinate, x[:noedges], nodes2edge, edge2element)

        # -- Reconstruct u_h as in Aposteriori --
        n_nodes = coordinate.shape[0]
        u_h = np.zeros((n_nodes, 2))
        supp_area = np.zeros(n_nodes)

        for j in range(element.shape[0]):
            nodes = element[j]
            coords = coordinate[nodes]
            area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))
            supp_area[nodes] += area / 3.0
            Pe = pEval[3 * j : 3 * j + 3]
            W = np.array([[4, 1, 1], [1, 4, 1], [1, 1, 4]]) / 36.0
            u_local = Pe.T @ W
            u_h[nodes] += area * u_local.T

        temp_uh = u_h / supp_area[:, None]
        u_h = tangent(coordinate, dirichlet, temp_uh)

        eta_T_local = np.zeros(element.shape[0])
        for j in range(element.shape[0]):
            nodes = element[j]
            coords = coordinate[nodes]
            area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))

            uh_local = u_h[nodes]
            Pe = pEval[3 * j : 3 * j + 3]
            W = np.array([[4, 1, 1], [1, 4, 1], [1, 1, 4]]) / 6.0
            diff = W @ uh_local - Pe
            square_diff = np.sum(diff**2, axis=1)
            eta_T_local[j] = np.sqrt(area * np.sum(square_diff) / 6.0)

        eta = np.sqrt(np.sum(eta_T_local**2))
        print(f"Estimated error: {eta:.3e}")

        if eta < tol:
            print("Tolerance reached. Stopping refinement.")
            break

        marked = mark_elements(eta_T_local)
        coordinate, element = refine_mesh(coordinate, element, marked)

    return coordinate, element

# -- Main Call --
def main():
    coordinate = np.loadtxt("coordinate.dat")
    element = np.loadtxt("element.dat", dtype=int) - 1
    dirichlet = np.loadtxt("dirichlet.dat", dtype=int) - 1
    try:
        Neumann = np.loadtxt("Neumann.dat", dtype=int) - 1
    except:
        Neumann = np.array([])

    coordinate, element = adaptive_RT(coordinate, element, dirichlet, Neumann, tol=1e-3, max_cycles=10)

if __name__ == "__main__":
    main()
