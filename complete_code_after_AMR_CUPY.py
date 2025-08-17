import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm # For progress bar

# ==============================================================================
# 0. CUDA (GPU) SETUP
# ==============================================================================
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    from cupyx.scipy.sparse.linalg import spsolve as spsolve_gpu
    USE_GPU = True
    print("✅ CuPy found. Using GPU for acceleration.")
except ImportError:
    USE_GPU = False
    print("⚠️ CuPy not found. Falling back to CPU (NumPy/SciPy).")

# ==============================================================================
# 1. MESH AND GEOMETRIC DATA STRUCTURES
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
    """Computes geometric data structures on the CPU."""
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
# 2. FEM ASSEMBLY AND SOLVER
# ==============================================================================

# --- CPU Functions ---

def stimaB(coords):
    P1, P2, P3 = coords[0, :], coords[1, :], coords[2, :]
    N = np.zeros((6, 3))
    N[0:2, 1], N[0:2, 2] = P1 - P2, P3 - P1
    N[2:4, 0], N[2:4, 2] = P2 - P1, P3 - P2
    N[4:6, 0], N[4:6, 1] = P1 - P3, P2 - P3
    C_diag = np.array([np.linalg.norm(P2 - P3), np.linalg.norm(P3 - P1), np.linalg.norm(P1 - P2)])
    C = np.diag(C_diag)
    M = sp.diags([np.ones(2), np.ones(4), 2*np.ones(6), np.ones(4), np.ones(2)], [-4, -2, 0, 2, 4], shape=(6,6)).toarray()
    area_det = np.linalg.det(np.vstack([coords.T, np.ones(3)]))
    return (C @ N.T @ M @ N @ C) / (24 * area_det)

def assemble_global_matrices(coordinate, element, noedges, nodes2edge, edge2element):
    NT = element.shape[0]
    A_global = sp.lil_matrix((noedges, noedges))
    B_global = sp.lil_matrix((noedges, NT))
    for j in range(NT):
        elem_nodes = element[j, :]
        coords = coordinate[elem_nodes, :]
        I = np.array([nodes2edge[elem_nodes[i], elem_nodes[(i+1)%3]] - 1 for i in range(3)])
        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1: signum[k] = -1
        B_local = stimaB(coords)
        A_global[np.ix_(I, I)] += np.diag(signum) @ B_local @ np.diag(signum)
        n_vecs = coords[[2, 0, 1], :] - coords[[1, 2, 0], :]
        B_global[I, j] = signum * np.linalg.norm(n_vecs, axis=1)
    return A_global.tocsr(), B_global.tocsr()

# --- GPU Functions ---

def stimaB_gpu(coords):
    NT = coords.shape[0]
    P1, P2, P3 = coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]
    N = cp.zeros((NT, 6, 3))
    N[:, 0:2, 1], N[:, 0:2, 2] = P1 - P2, P3 - P1
    N[:, 2:4, 0], N[:, 2:4, 2] = P2 - P1, P3 - P2
    N[:, 4:6, 0], N[:, 4:6, 1] = P1 - P3, P2 - P3
    C_diag = cp.linalg.norm(cp.stack([P2 - P3, P3 - P1, P1 - P2], axis=1), axis=2)
    C = cp.einsum('ij,jk->ijk', C_diag, cp.eye(3))
    M = cp.asarray(sp.diags([[1,1],[1,1,1,1],[2,2,2,2,2,2],[1,1,1,1],[1,1]], [-4,-2,0,2,4], shape=(6,6)).toarray())
    area_det = cp.linalg.det(cp.concatenate([coords, cp.ones((NT, 3, 1))], axis=2))
    B_local = cp.einsum('nij,nkj,kl,nlm,nmo,n->nio', C, N, M, N, C, 1.0 / (24 * area_det))
    return B_local

def assemble_global_matrices_gpu(coordinate, element, noedges, nodes2edge, edge2element):
    NT = element.shape[0]
    coords = coordinate[element]
    I = nodes2edge[element[:, [0, 1, 2]], element[:, [1, 2, 0]]] - 1
    elem_indices = cp.arange(NT).reshape(NT, 1)
    signum = cp.where(edge2element[I, 3] == elem_indices + 1, -1.0, 1.0)
    B_local = stimaB_gpu(coords)
    signum_matrices = cp.einsum('ni,ij->nij', signum, cp.eye(3, dtype=signum.dtype))
    A_local = cp.einsum('nij,njk,nkl->nil', signum_matrices, B_local, signum_matrices)
    row_indices, col_indices = cp.repeat(I, 3).ravel(), cp.tile(I, (1, 3)).ravel()
    data = A_local.ravel()
    A_global = cusp.coo_matrix((data, (row_indices, col_indices)), shape=(noedges, noedges)).tocsr()
    n_vecs = coords[:, [2, 0, 1], :] - coords[:, [1, 2, 0], :]
    B_data = (signum * cp.linalg.norm(n_vecs, axis=2)).ravel()
    B_row_indices, B_col_indices = I.ravel(), cp.repeat(cp.arange(NT), 3)
    B_global = cusp.coo_matrix((B_data, (B_row_indices, B_col_indices)), shape=(noedges, NT)).tocsr()
    return A_global, B_global

# ==============================================================================
# 3. ADAPTIVE MESH REFINEMENT
# ==============================================================================

def a_posteriori_estimator(element, coordinate, p_h_vectors):
    n_nodes = coordinate.shape[0]
    u_h_avg, supp_area = np.zeros((n_nodes, 2)), np.zeros(n_nodes)
    for j, elem in enumerate(element):
        area = 0.5 * abs(np.linalg.det(np.vstack([coordinate[elem].T, np.ones(3)])))
        supp_area[elem] += area / 3.0
        local_p = p_h_vectors[3*j : 3*j+3, :]
        u_h_avg[elem] += area * (np.array([[2,1,1],[1,2,1],[1,1,2]]) @ local_p) / 12.0
    u_h_avg[supp_area > 0] /= supp_area[supp_area > 0][:, None]
    eta_T = np.zeros(len(element))
    for j, elem in enumerate(element):
        area = 0.5 * abs(np.linalg.det(np.vstack([coordinate[elem].T, np.ones(3)])))
        local_p = p_h_vectors[3*j : 3*j+3, :]
        diff = local_p - u_h_avg[elem]
        integral = area * np.sum((np.array([[2,1,1],[1,2,1],[1,1,2]]) @ (diff*diff))) / 6.0
        eta_T[j] = np.sqrt(abs(integral))
    return eta_T

def a_posteriori_estimator_gpu(element, coordinate, p_h_vectors):
    """GPU-accelerated a posteriori error estimator."""
    NT, n_nodes = element.shape[0], coordinate.shape[0]
    
    coords = coordinate[element]
    areas = 0.5 * abs(cp.linalg.det(cp.concatenate([coords, cp.ones((NT, 3, 1))], axis=2)))

    u_h_avg = cp.zeros((n_nodes, 2), dtype=cp.float64)
    supp_area = cp.zeros(n_nodes, dtype=cp.float64)
    
    # --- WORKAROUND IMPLEMENTATION ---
    # The '.at' method is broken in your environment.
    # The following code uses cp.bincount as an alternative way to perform
    # the required atomic scatter-add operations.
    
    # 1. Calculate supp_area using bincount
    indices = element.ravel()
    area_contrib_vals = cp.tile(areas / 3.0, 3)
    # Use bincount to sum contributions for each node
    supp_area_update = cp.bincount(indices, weights=area_contrib_vals, minlength=n_nodes)
    supp_area += supp_area_update
    
    # 2. Calculate u_h_avg using bincount
    weight_matrix = cp.asarray([[2,1,1],[1,2,1],[1,1,2]], dtype=cp.float64) / 12.0
    local_p = p_h_vectors.reshape(NT, 3, 2)
    weighted_p = cp.einsum('ij,njk->nik', weight_matrix, local_p)
    avg_contrib = areas[:, None, None] * weighted_p

    # Ravel the contributions for x and y components separately
    avg_contrib_x = avg_contrib[:, :, 0].ravel()
    avg_contrib_y = avg_contrib[:, :, 1].ravel()

    # Perform bincount for each component
    u_h_avg_x = cp.bincount(indices, weights=avg_contrib_x, minlength=n_nodes)
    u_h_avg_y = cp.bincount(indices, weights=avg_contrib_y, minlength=n_nodes)
    
    # Combine the results back into u_h_avg
    u_h_avg += cp.stack([u_h_avg_x, u_h_avg_y], axis=1)

    # --- END OF WORKAROUND ---

    supp_area[supp_area == 0] = 1
    u_h_avg /= supp_area[:, None]

    Ap_h_local = u_h_avg[element]
    diff = local_p - Ap_h_local
    
    integrand_matrix = cp.asarray([[2,1,1],[1,2,1],[1,1,2]], dtype=cp.float64) / 6.0
    # Perform the element-wise product, batched matrix multiplication,
    # and sum for each element all in one efficient step.
    integral_vals = cp.einsum('ij,njk->n', integrand_matrix, diff * diff)
    integral = areas * integral_vals
    eta_T = cp.sqrt(abs(integral))
    
    return eta_T

def mark_elements(eta_T):
    """Marks elements for refinement. Works on both CPU and GPU arrays."""
    xp = cp.get_array_module(eta_T) if 'cupy' in str(type(eta_T)) else np
    max_eta = xp.max(eta_T)
    return xp.where(eta_T >= 0.5 * max_eta)[0]

def refine_mesh(coordinate, element, dirichlet, neumann, marked_elements):
    """Refines the mesh. This logic is highly sequential and remains on the CPU."""
    marked_set = set(marked_elements)
    
    while True:
        newly_marked = set()
        for elem_idx in marked_set:
            if elem_idx >= len(element): continue
            for i in range(3):
                p1, p2 = element[elem_idx, i], element[elem_idx, (i+1)%3]
                for other_idx, other_elem in enumerate(element):
                    if other_idx != elem_idx and other_idx not in marked_set:
                        other_nodes = set(other_elem)
                        if p1 in other_nodes and p2 in other_nodes:
                            newly_marked.add(other_idx)
        if not newly_marked: break
        marked_set.update(newly_marked)

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
            m12, m23, m31 = get_midpoint(p1,p2), get_midpoint(p2,p3), get_midpoint(p3,p1)
            new_elements.extend([[p1,m12,m31], [m12,p2,m23], [m31,m23,p3], [m12,m23,m31]])
        else:
            new_elements.append(list(elem))
            
    def refine_boundary(boundary_edges):
        new_boundary = []
        if boundary_edges.shape[0] == 0: return boundary_edges
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
# 4. POST-PROCESSING AND VISUALIZATION
# ==============================================================================

def show_displacement(elements, coordinates, u_h):
    """Visualizes the piecewise constant displacement field u_h."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    vmin, vmax = np.min(u_h), np.max(u_h)
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        ax.plot_trisurf(verts[:,0], verts[:,1], np.full(3, u_h[j]), vmin=vmin, vmax=vmax, cmap='viridis', edgecolor='k', linewidth=0.2)
    ax.set_title('Displacement u_h'); ax.view_init(30, -120); plt.show()

def flux_eb_eval(coordinate, element, noedges, nodes2edge, edge2element, p_sol):
    """Computes the flux vector p_h (CPU version)."""
    NT = element.shape[0]
    p_h = np.zeros((3 * NT, 2))
    for j in range(NT):
        elem_nodes = element[j, :]
        coords = coordinate[elem_nodes, :]
        I = np.array([nodes2edge[elem_nodes[i], elem_nodes[(i+1)%3]]-1 for i in range(3)])
        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1: signum[k] = -1
        area = 0.5 * np.linalg.det(np.vstack([coords.T, np.ones(3)]))
        centroid = np.mean(coords, axis=0)
        flux_vec = np.zeros(2)
        for i in range(3):
            edge_len = np.linalg.norm(coords[(i+1)%3] - coords[(i+2)%3])
            psi = signum[i] * edge_len / (2*area) * (centroid-coords[i])
            flux_vec += p_sol[I[i]] * psi
        p_h[3*j : 3*j+3, :] = np.tile(flux_vec, (3, 1))
    return p_h
    
def flux_eb_eval_gpu(coordinate, element, noedges, nodes2edge, edge2element, p_sol):
    """GPU-accelerated flux vector computation."""
    NT = element.shape[0]
    coords = coordinate[element]
    areas = 0.5 * cp.linalg.det(cp.concatenate([coords, cp.ones((NT, 3, 1))], axis=2))
    centroids = cp.mean(coords, axis=1)
    I = nodes2edge[element[:, [0, 1, 2]], element[:, [1, 2, 0]]] - 1
    elem_indices = cp.arange(NT).reshape(NT, 1)
    signum = cp.where(edge2element[I, 3] == elem_indices + 1, -1.0, 1.0)
    edge_lens = cp.linalg.norm(coords[:, [1, 2, 0], :] - coords[:, [2, 0, 1], :], axis=2)
    psi = (signum*edge_lens/(2*areas[:,None]))[:,:,None] * (centroids[:,None,:]-coords)
    flux_vecs = cp.sum(p_sol[I][:, :, None] * psi, axis=1)
    p_h = cp.tile(flux_vecs, (1, 3)).reshape(3 * NT, 2)
    return p_h

def show_flux(elements, coordinates, p_h):
    """Visualizes the x and y components of the flux vector p_h."""
    if 'cupy' in str(type(p_h)): p_h = cp.asnumpy(p_h)
    px, py = p_h[2::3, 0], p_h[2::3, 1]
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    vmin_x, vmax_x = np.min(px), np.max(px)
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        ax1.plot_trisurf(verts[:,0], verts[:,1], np.full(3, px[j]), vmin=vmin_x, vmax=vmax_x, cmap='coolwarm', edgecolor='k', linewidth=0.2)
    ax1.set_title('Flux Component p_x'); ax1.view_init(30, -120)
    ax2 = fig.add_subplot(122, projection='3d')
    vmin_y, vmax_y = np.min(py), np.max(py)
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        ax2.plot_trisurf(verts[:,0], verts[:,1], np.full(3, py[j]), vmin=vmin_y, vmax=vmax_y, cmap='coolwarm', edgecolor='k', linewidth=0.2)
    ax2.set_title('Flux Component p_y'); ax2.view_init(30, -120)
    plt.tight_layout(); plt.show()

# ==============================================================================
# 5. MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    NUM_REFINEMENTS = 5
    coordinate, element, dirichlet, neumann = load_mesh_files()

    for step in tqdm(range(NUM_REFINEMENTS), desc="AMR Steps"):
        
        nodes2element, nodes2edge, noedges, edge2element, edge_map = edge(coordinate, element)
        NT = element.shape[0] # Define NT here so it's available for the whole loop
        
        # No internal heat source
        f_source = lambda x: 0.0

        # Dirichlet boundary condition: Fixed temperatures
        def u_D_func(x):
            if np.isclose(x[0], -2.0):  # Left edge
                return 100.0
            elif np.isclose(x[0], 2.0):  # Right edge
                return 0.0
            return 0.0

        # Neumann boundary condition: Insulated edges (zero heat flux)
        g_func = lambda x, n: 0.0
        def g_func(x, n):
            x = np.array(x)
            r, phi = np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])
            if phi < 0: phi += 2 * np.pi
            val_vec = (2/3) * r**(-1/3) * np.array([-np.sin(phi/3), np.cos(phi/3)])
            return np.dot(val_vec, n)

        if USE_GPU:
            coordinate_d, element_d = cp.asarray(coordinate), cp.asarray(element)
            nodes2edge_d = cp.asarray(nodes2edge.toarray())
            edge2element_d = cp.asarray(edge2element)
            A, C_T = assemble_global_matrices_gpu(coordinate_d, element_d, noedges, nodes2edge_d, edge2element_d)
            b, x_bc = cp.zeros(noedges + NT), cp.zeros(noedges + NT)
        else:
            A, C_T = assemble_global_matrices(coordinate, element, noedges, nodes2edge, edge2element)
            b, x_bc = np.zeros(noedges + NT), np.zeros(noedges + NT)
        
        coords_all = coordinate[element]
        areas = 0.5 * np.linalg.det(np.concatenate([coords_all, np.ones((NT, 3, 1))], axis=2))
        centroids = np.mean(coords_all, axis=1)
        b_f_cpu = -areas * np.array([f_source(c) for c in centroids])
        
        if USE_GPU: b[noedges:] = cp.asarray(b_f_cpu)
        else: b[noedges:] = b_f_cpu
        
        dirichlet_edges_idx = [edge_map[tuple(sorted(d))] for d in dirichlet]
        neumann_edges_idx = [edge_map[tuple(sorted(n))] for n in neumann]
        
        for idx, d_edge in enumerate(dirichlet):
            midpoint = np.mean(coordinate[d_edge,:], axis=0)
            edge_len = np.linalg.norm(coordinate[d_edge[0],:] - coordinate[d_edge[1],:])
            b[dirichlet_edges_idx[idx]] += u_D_func(midpoint) * edge_len

        for idx, n_edge in enumerate(neumann):
            midpoint = np.mean(coordinate[n_edge,:], axis=0)
            vec = coordinate[n_edge[1],:] - coordinate[n_edge[0],:]
            normal = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
            elem_idx = edge2element[neumann_edges_idx[idx], 2] - 1
            v3_idx = [p for p in element[elem_idx] if p not in n_edge][0]
            if np.dot(normal, coordinate[v3_idx] - coordinate[n_edge[0]]) > 0: normal = -normal
            x_bc[neumann_edges_idx[idx]] = g_func(midpoint, normal)
            
        if USE_GPU:
            SystemMatrix = cusp.vstack([cusp.hstack([A,C_T]), cusp.hstack([C_T.T, cusp.csr_matrix((NT,NT),dtype=cp.float64)])])
        else:
            SystemMatrix = sp.vstack([sp.hstack([A,C_T]), sp.hstack([C_T.T, sp.csr_matrix((NT,NT))])])
        
        b -= SystemMatrix @ x_bc
        free_dofs = [i for i in range(noedges) if i not in set(dirichlet_edges_idx+neumann_edges_idx)] + list(range(noedges, noedges+NT-1))
        
        SystemMatrix_csr = SystemMatrix.tocsr()
        
        if USE_GPU:
            A_free = SystemMatrix_csr[cp.ix_(free_dofs, free_dofs)]
            sol_free = spsolve_gpu(A_free, b[free_dofs])
        else:
            A_free = SystemMatrix_csr[np.ix_(free_dofs, free_dofs)]
            sol_free = spsolve(A_free, b[free_dofs])
        
        solution = x_bc.copy()
        solution[free_dofs] = sol_free
        p_sol, u_sol = solution[:noedges], solution[noedges:]
        u_sol[-1] = 0

        if USE_GPU:
            p_h_vectors = flux_eb_eval_gpu(coordinate_d, element_d, noedges, nodes2edge_d, edge2element_d, p_sol)
            eta_T = a_posteriori_estimator_gpu(element_d, coordinate_d, p_h_vectors)
        else:
            p_h_vectors = flux_eb_eval(coordinate, element, noedges, nodes2edge, edge2element, p_sol)
            eta_T = a_posteriori_estimator(coordinate, element, p_h_vectors)

        eta_T_cpu = cp.asnumpy(eta_T) if USE_GPU else eta_T
        total_error = np.sqrt(np.sum(eta_T_cpu**2))

        print(f"\nStep {step+1}/{NUM_REFINEMENTS}:")
        print(f"  Elements: {NT}, Edges: {noedges}")
        print(f"  Total Estimated Error (eta_A): {total_error:.4f}")

        marked_elements = mark_elements(eta_T_cpu)
        print(f"  Marked {len(marked_elements)} elements for refinement.")

        if step < NUM_REFINEMENTS - 1:
            marked_elements_np = cp.asnumpy(marked_elements) if 'cupy' in str(type(marked_elements)) else marked_elements
            coordinate, element, dirichlet, neumann = refine_mesh(coordinate, element, dirichlet, neumann, marked_elements_np)
    
    print("\nDisplaying final results on the adaptively refined mesh...")
    if USE_GPU:
        u_sol_cpu, p_h_vectors_cpu = cp.asnumpy(u_sol), cp.asnumpy(p_h_vectors)
    else:
        u_sol_cpu, p_h_vectors_cpu = u_sol, p_h_vectors
    
    show_displacement(element, coordinate, u_sol_cpu)
    show_flux(element, coordinate, p_h_vectors_cpu)