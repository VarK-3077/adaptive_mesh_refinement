import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm # For progress bar

# ==============================================================================
# 1. MESH AND GEOMETRIC DATA STRUCTURES (from edge.py)
# ==============================================================================

def load_mesh_files():
    """Loads mesh data from standard .dat files."""
    coordinates = np.loadtxt('coordinate.dat')
    element = np.loadtxt('element.dat', dtype=int) - 1
    try:
        dirichlet = np.loadtxt('Dirichlet.dat', dtype=int) - 1
        if dirichlet.ndim == 1: dirichlet = dirichlet.reshape(1, 2)
    except IOError:
        dirichlet = np.empty((0, 2), dtype=int)
    try:
        neumann = np.loadtxt('Neumann.dat', dtype=int) - 1
        if neumann.ndim == 1: neumann = neumann.reshape(1, 2)
    except IOError:
        neumann = np.empty((0, 2), dtype=int)
    return coordinates, element, dirichlet, neumann

def edge(coordinate, element):
    """
    Computes geometric data structures: nodes2edge, edge2element, etc. [cite: 974]
    """
    n_nodes = coordinate.shape[0]
    nodes2element = sp.lil_matrix((n_nodes, n_nodes), dtype=int)
    for idx, elem in enumerate(element):
        for i in range(3):
            nodes2element[elem[i], elem[(i + 1) % 3]] = idx + 1

    edge_map = {}
    nodes2edge = sp.lil_matrix((n_nodes, n_nodes), dtype=int)
    edge_count = 0
    for elem in element:
        for i in range(3):
            p1, p2 = sorted((elem[i], elem[(i+1)%3]))
            if (p1, p2) not in edge_map:
                edge_map[(p1, p2)] = edge_count
                nodes2edge[p1, p2] = edge_count + 1
                edge_count += 1
    nodes2edge = nodes2edge + nodes2edge.T

    noedges = len(edge_map)
    edge2element = np.zeros((noedges, 4), dtype=int)
    for j, elem in enumerate(element):
        for i in range(3):
            p1, p2 = elem[i], elem[(i + 1) % 3]
            edge_idx = edge_map[tuple(sorted((p1, p2)))]
            if edge2element[edge_idx, 0] == 0:
                edge2element[edge_idx, 0] = p1 + 1
                edge2element[edge_idx, 1] = p2 + 1
                edge2element[edge_idx, 2] = j + 1
            else:
                edge2element[edge_idx, 3] = j + 1
            if nodes2element[p1, p2] != j + 1:
                 T_plus, T_minus = edge2element[edge_idx, 3], edge2element[edge_idx, 2]
                 edge2element[edge_idx, 2], edge2element[edge_idx, 3] = T_plus, T_minus
    return nodes2element, nodes2edge, noedges, edge2element, edge_map

# ==============================================================================
# 2. FEM ASSEMBLY AND SOLVER (from EBfem.py)
# ==============================================================================

def stimaB(coords):
    P1, P2, P3 = coords[0, :], coords[1, :], coords[2, :]
    N = np.zeros((6, 3))
    N[0:2, 1], N[0:2, 2] = P1 - P2, P3 - P1
    N[2:4, 0], N[2:4, 2] = P2 - P1, P3 - P2
    N[4:6, 0], N[4:6, 1] = P1 - P3, P2 - P3
    C_diag = np.array([np.linalg.norm(P2 - P3), np.linalg.norm(P3 - P1), np.linalg.norm(P1 - P2)])
    C = np.diag(C_diag)
    M = sp.diags([np.ones(2), np.ones(4), 2*np.ones(6), np.ones(4), np.ones(2)], [-4, -2, 0, 2, 4], shape=(6,6)).toarray()
    area_det = np.linalg.det(np.vstack([np.ones(3), coords.T]))
    return (C @ N.T @ M @ N @ C) / (24 * area_det)

def assemble_global_matrices(coordinate, element, noedges, nodes2edge, edge2element):
    """Assembles the global stiffness matrices by looping through elements. [cite: 1145]"""
    NT = element.shape[0]
    A_global = sp.lil_matrix((noedges, noedges))
    B_global = sp.lil_matrix((noedges, NT))
    for j in range(NT):
        elem_nodes = element[j, :]
        coords = coordinate[elem_nodes, :]
        I = np.array([nodes2edge[elem_nodes[i], elem_nodes[(i+1)%3]] - 1 for i in range(3)])
        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1:
                signum[k] = -1
        B_local = stimaB(coords)
        A_global[np.ix_(I, I)] += np.diag(signum) @ B_local @ np.diag(signum)
        n_vecs = coords[[2, 0, 1], :] - coords[[1, 2, 0], :]
        B_global[I, j] = signum * np.linalg.norm(n_vecs, axis=1)
    return A_global.tocsr(), B_global.tocsr()

# ==============================================================================
# 3. ADAPTIVE MESH REFINEMENT FUNCTIONS
# ==============================================================================

def a_posteriori_estimator(element, coordinate, p_h_vectors):
    """
    Computes the local error indicator eta_T for each element.
    This implements the averaging technique from Section 8 of the paper. 
    """
    n_nodes = coordinate.shape[0]
    u_h_avg = np.zeros((n_nodes, 2))
    supp_area = np.zeros(n_nodes)
    
    # Average the elemental flux p_h to the nodes
    for j, elem in enumerate(element):
        area = 0.5 * abs(np.linalg.det(np.vstack([np.ones(3), coordinate[elem].T])))
        supp_area[elem] += area / 3.0
        local_p = p_h_vectors[3*j : 3*j+3, :]
        # Weight matrix for averaging
        u_h_avg[elem] += area * (np.array([[2,1,1],[1,2,1],[1,1,2]]) @ local_p) / 12.0

    u_h_avg /= supp_area[:, None]

    # Compute local error indicators
    eta_T = np.zeros(len(element))
    for j, elem in enumerate(element):
        area = 0.5 * abs(np.linalg.det(np.vstack([np.ones(3), coordinate[elem].T])))
        local_p = p_h_vectors[3*j : 3*j+3, :]
        Ap_h_local = u_h_avg[elem]
        
        # Integral of ||p_h - Ap_h||^2 over the element T
        diff = local_p - Ap_h_local
        integral = area * np.sum( (np.array([[2,1,1],[1,2,1],[1,1,2]]) @ (diff * Ap_h_local) ) ) / 6.0
        eta_T[j] = np.sqrt(abs(integral))

    return eta_T

def mark_elements(eta_T):
    """
    Marks elements for refinement based on the local error indicators.
    Implements the marking strategy from Section 9.3 of the paper. 
    """
    max_eta = np.max(eta_T)
    return np.where(eta_T >= 0.5 * max_eta)[0]

def refine_mesh(coordinate, element, dirichlet, neumann, marked_elements):
    """
    Refines the mesh using a red refinement strategy for marked elements.
    Ensures conformity by refining neighbors to avoid hanging nodes.
    """
    marked_set = set(marked_elements)
    
    # Closure: Mark neighbors of marked elements to avoid hanging nodes
    while True:
        newly_marked = set()
        for elem_idx in marked_set:
            for i in range(3):
                p1, p2 = element[elem_idx, i], element[elem_idx, (i+1)%3]
                # Find neighbor
                for other_idx, other_elem in enumerate(element):
                    if other_idx not in marked_set and (p1 in other_elem and p2 in other_elem):
                        newly_marked.add(other_idx)
        if not newly_marked:
            break
        marked_set.update(newly_marked)

    # Actual refinement
    coord_list = list(coordinate)
    new_elements = []
    edge_to_midpoint = {}

    def get_midpoint(p1_idx, p2_idx):
        edge = tuple(sorted((p1_idx, p2_idx)))
        if edge not in edge_to_midpoint:
            mid_coord = (coordinate[p1_idx] + coordinate[p2_idx]) / 2.0
            coord_list.append(mid_coord)
            new_idx = len(coord_list) - 1
            edge_to_midpoint[edge] = new_idx
        return edge_to_midpoint[edge]

    for elem_idx, elem in enumerate(element):
        if elem_idx in marked_set:
            p1, p2, p3 = elem
            m12 = get_midpoint(p1, p2)
            m23 = get_midpoint(p2, p3)
            m31 = get_midpoint(p3, p1)
            new_elements.extend([
                [p1, m12, m31],
                [m12, p2, m23],
                [m31, m23, p3],
                [m12, m23, m31]
            ])
        else:
            new_elements.append(elem)
            
    # Refine boundary edges
    def refine_boundary(boundary_edges):
        new_boundary = []
        for p1, p2 in boundary_edges:
            edge = tuple(sorted((p1, p2)))
            if edge in edge_to_midpoint:
                mid_idx = edge_to_midpoint[edge]
                new_boundary.extend([[p1, mid_idx], [mid_idx, p2]])
            else:
                new_boundary.append([p1, p2])
        return np.array(new_boundary, dtype=int)

    return (np.array(coord_list), np.array(new_elements, dtype=int),
            refine_boundary(dirichlet), refine_boundary(neumann))

# ==============================================================================
# 4. POST-PROCESSING AND VISUALIZATION (from postproce.py)
# ==============================================================================

def show_displacement(elements, coordinates, u_h):
    """Visualizes the piecewise constant displacement field u_h."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    vmin, vmax = np.min(u_h), np.max(u_h)
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        z_values = np.full(3, u_h[j])
        ax.plot_trisurf(verts[:, 0], verts[:, 1], z_values, vmin=vmin, vmax=vmax, cmap='viridis', edgecolor='k', linewidth=0.2)
    ax.set_title('Displacement u_h')
    ax.view_init(30, -120)
    plt.savefig('displacment.png')

def flux_eb_eval(coordinate, element, noedges, nodes2edge, edge2element, p_sol):
    """Computes the flux vector p_h inside each element from the solution vector. [cite: 1402]"""
    NT = element.shape[0]
    p_h = np.zeros((3 * NT, 2))
    for j in range(NT):
        elem_nodes = element[j, :]
        coords = coordinate[elem_nodes, :]
        I = np.array([nodes2edge[elem_nodes[i], elem_nodes[(i+1)%3]] - 1 for i in range(3)])
        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1:
                signum[k] = -1
        area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))
        centroid = np.mean(coords, axis=0)
        flux_vec = np.zeros(2)
        for i in range(3):
            edge_len = np.linalg.norm(coords[(i+1)%3] - coords[(i+2)%3])
            opposite_vertex = coords[i]
            psi = signum[i] * edge_len / (2 * area) * (centroid - opposite_vertex)
            flux_vec += p_sol[I[i]] * psi
        p_h[3*j : 3*j+3, :] = np.tile(flux_vec, (3, 1))
    return p_h

def show_flux(elements, coordinates, p_h):
    """Visualizes the x and y components of the piecewise-constant flux vector p_h."""
    px_elemental, py_elemental = p_h[2::3, 0], p_h[2::3, 1]
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    vmin_x, vmax_x = np.min(px_elemental), np.max(px_elemental)
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        z_values = np.full(3, px_elemental[j])
        ax1.plot_trisurf(verts[:, 0], verts[:, 1], z_values, vmin=vmin_x, vmax=vmax_x, cmap='coolwarm', edgecolor='k', linewidth=0.2)
    ax1.set_title('Flux Component p_x'), ax1.view_init(30, -120)
    ax2 = fig.add_subplot(122, projection='3d')
    vmin_y, vmax_y = np.min(py_elemental), np.max(py_elemental)
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        z_values = np.full(3, py_elemental[j])
        ax2.plot_trisurf(verts[:, 0], verts[:, 1], z_values, vmin=vmin_y, vmax=vmax_y, cmap='coolwarm', edgecolor='k', linewidth=0.2)
    ax2.set_title('Flux Component p_y'), ax2.view_init(30, -120)
    plt.tight_layout(), plt.show()

# ==============================================================================
# 5. MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- AMR Setup ---
    NUM_REFINEMENTS = 5
    
    # --- Initial Mesh Load ---
    coordinate, element, dirichlet, neumann = load_mesh_files()

    # --- Adaptive Loop ---
    for step in tqdm(range(NUM_REFINEMENTS), desc="AMR Steps"):
        
        # --- 1. Build Geometric Data ---
        nodes2element, nodes2edge, noedges, edge2element, edge_map = edge(coordinate, element)
        
        # --- 2. Define the Problem (L-shaped domain, Section 9.1) ---
        f_source = lambda x: 0.0
        u_D_func = lambda x: 0.0
        def g_func(x, n):
            r, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
            if phi < 0: phi += 2 * np.pi
            val_vec = (2/3) * r**(-1/3) * np.array([-np.sin(phi/3), np.cos(phi/3)])
            return np.dot(val_vec, n)

        # --- 3. Assemble and Solve (SOLVE) ---
        A, C_T = assemble_global_matrices(coordinate, element, noedges, nodes2edge, edge2element)
        NT = element.shape[0]
        b = np.zeros(noedges + NT)
        b_f = np.zeros(NT)
        for j in range(NT):
            coords = coordinate[element[j, :], :]
            area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))
            b_f[j] = -area * f_source(np.mean(coords, axis=0))
        b[noedges:] = b_f

        x_bc = np.zeros(noedges + NT)
        dirichlet_edges_idx = [edge_map[tuple(sorted(d))] for d in dirichlet]
        neumann_edges_idx = [edge_map[tuple(sorted(n))] for n in neumann]
        
        for idx, d_edge in enumerate(dirichlet):
            midpoint = np.mean(coordinate[d_edge, :], axis=0)
            edge_len = np.linalg.norm(coordinate[d_edge[0], :] - coordinate[d_edge[1], :])
            b[dirichlet_edges_idx[idx]] += u_D_func(midpoint) * edge_len

        for idx, n_edge in enumerate(neumann):
            midpoint = np.mean(coordinate[n_edge, :], axis=0)
            vec = coordinate[n_edge[1], :] - coordinate[n_edge[0], :]
            normal = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
            elem_idx_T_plus = edge2element[neumann_edges_idx[idx], 2] - 1
            v3_idx = [p for p in element[elem_idx_T_plus] if p not in n_edge][0]
            if np.dot(normal, coordinate[v3_idx] - coordinate[n_edge[0]]) > 0: normal = -normal
            x_bc[neumann_edges_idx[idx]] = g_func(midpoint, normal)
            
        SystemMatrix = sp.vstack([sp.hstack([A, C_T]), sp.hstack([C_T.T, sp.csr_matrix((NT, NT))])])
        b -= SystemMatrix @ x_bc
        
        free_edges_idx = [i for i in range(noedges) if i not in set(dirichlet_edges_idx + neumann_edges_idx)]
        free_dofs = free_edges_idx + list(range(noedges, noedges + NT - 1))
        
        SystemMatrix_csr = SystemMatrix.tocsr()
        A_free = SystemMatrix_csr[np.ix_(free_dofs, free_dofs)]
        b_free = b[free_dofs]
        
        sol_free = spsolve(A_free, b_free)
        
        solution = np.copy(x_bc)
        solution[free_dofs] = sol_free
        p_sol, u_sol = solution[:noedges], solution[noedges:]
        u_sol[-1] = 0

        # --- 4. Post-Process for Estimation (ESTIMATE) ---
        p_h_vectors = flux_eb_eval(coordinate, element, noedges, nodes2edge, edge2element, p_sol)
        eta_T = a_posteriori_estimator(element, coordinate, p_h_vectors)

        print(f"\nStep {step+1}/{NUM_REFINEMENTS}:")
        print(f"  Elements: {NT}, Edges: {noedges}")
        print(f"  Total Estimated Error (eta_A): {np.sqrt(np.sum(eta_T**2)):.4f}")

        # --- 5. Mark Elements (MARK) ---
        marked_elements = mark_elements(eta_T)
        print(f"  Marked {len(marked_elements)} elements for refinement.")

        # --- 6. Refine Mesh (REFINE) ---
        if step < NUM_REFINEMENTS - 1:
            coordinate, element, dirichlet, neumann = refine_mesh(coordinate, element, dirichlet, neumann, marked_elements)
    
    # --- 7. Final Visualization ---
    print("\nDisplaying final results on the adaptively refined mesh...")
    show_displacement(element, coordinate, u_sol)
    show_flux(element, coordinate, p_h_vectors)