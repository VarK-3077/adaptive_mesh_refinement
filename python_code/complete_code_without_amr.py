import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. MESH AND GEOMETRIC DATA STRUCTURES (from edge.py)
# These functions load the mesh from files and compute the essential
# connectivity matrices described in Section 3 of the paper. [cite: 114]
# ==============================================================================

def load_mesh_files():
    """Loads mesh data from standard .dat files."""
    coordinates = np.loadtxt('coordinate.dat')
    # Element indices are 1-based in the file, convert to 0-based for Python
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
    Computes geometric data structures: nodes2edge, edge2element, etc.
    This corresponds to the 'edge.m' function described in Section 3.2 of the paper. [cite: 222]
    """
    n_nodes = coordinate.shape[0]
    nodes2element = sp.lil_matrix((n_nodes, n_nodes), dtype=int)
    for idx, elem in enumerate(element):
        for i in range(3):
            nodes2element[elem[i], elem[(i + 1) % 3]] = idx + 1 # 1-based element index

    edge_map = {}
    nodes2edge = sp.lil_matrix((n_nodes, n_nodes), dtype=int)
    edge_count = 0
    for elem in element:
        for i in range(3):
            p1, p2 = sorted((elem[i], elem[(i+1)%3]))
            if (p1, p2) not in edge_map:
                edge_map[(p1, p2)] = edge_count
                nodes2edge[p1, p2] = edge_count + 1 # 1-based edge index
                edge_count += 1
    
    nodes2edge = nodes2edge + nodes2edge.T

    noedges = len(edge_map)
    edge2element = np.zeros((noedges, 4), dtype=int)
    
    for j, elem in enumerate(element):
        for i in range(3):
            p1, p2 = elem[i], elem[(i + 1) % 3]
            edge_idx = edge_map[tuple(sorted((p1, p2)))]
            
            if edge2element[edge_idx, 0] == 0: # First time seeing this edge
                edge2element[edge_idx, 0] = p1 + 1 # 1-based node indices
                edge2element[edge_idx, 1] = p2 + 1
                edge2element[edge_idx, 2] = j + 1 # T+ element
            else: # Second time, this is the neighbor
                edge2element[edge_idx, 3] = j + 1 # T- element
                
            # Ensure consistent orientation (T+ is where the normal points from)
            if nodes2element[p1, p2] != j + 1:
                # This element is T- for the edge from p1 to p2, swap T+ and T-
                 T_plus = edge2element[edge_idx, 3]
                 T_minus = edge2element[edge_idx, 2]
                 edge2element[edge_idx, 2] = T_plus
                 edge2element[edge_idx, 3] = T_minus
                 
    return nodes2element, nodes2edge, noedges, edge2element

# ==============================================================================
# 2. FEM ASSEMBLY AND SOLVER (from EBfem.py)
# These functions assemble and solve the global linear system for the MFEM.
# ==============================================================================

def stimaB(coords):
    """
    Computes the local 3x3 stiffness matrix for RT0 elements on a single triangle.
    This is a corrected implementation based on the mathematical formulas in
    Section 4.2 of the paper, specifically Lemma 4.2.
    """
    # coords is a (3, 2) array with vertices P1, P2, P3
    P1, P2, P3 = coords[0, :], coords[1, :], coords[2, :]

    # Construct the 6x3 matrix N based on Lemma 4.2.
    # This matrix contains the coordinate differences between vertices.
    N = np.zeros((6, 3))
    N[0:2, 1] = P1 - P2
    N[0:2, 2] = P3 - P1
    N[2:4, 0] = P2 - P1
    N[2:4, 2] = P3 - P2
    N[4:6, 0] = P1 - P3
    N[4:6, 1] = P2 - P3

    # Construct the 3x3 diagonal matrix C. The diagonal entries are the lengths
    # of the edges opposite the vertices: |E1|, |E2|, |E3|.
    # This logic matches the paper's MATLAB code for C.
    C_diag = np.array([
        np.linalg.norm(P2 - P3),  # |E1|, opposite P1
        np.linalg.norm(P3 - P1),  # |E2|, opposite P2
        np.linalg.norm(P1 - P2)   # |E3|, opposite P3
    ])
    C = np.diag(C_diag)

    # Construct the 6x6 matrix M as defined in Lemma 4.2 
    M_diag_vals = [np.ones(2), np.ones(4), 2 * np.ones(6), np.ones(4), np.ones(2)]
    M = sp.diags(M_diag_vals, [-4, -2, 0, 2, 4], shape=(6, 6)).toarray()

    # Calculate the determinant term for the area, as in formula (4.2) [cite: 316]
    area_det = np.linalg.det(np.vstack([np.ones(3), coords.T]))

    # Calculate the local stiffness matrix B_local using the formula from Lemma 4.2[cite: 372].
    # With the corrected matrix shapes, the matmul operations are now valid.
    B_local = (C @ N.T @ M @ N @ C) / (24 * area_det)
    return B_local

def assemble_global_matrices(coordinate, element, noedges, nodes2edge, edge2element):
    """
    Assembles the global stiffness matrices by looping through elements.
    This implements the procedure described in Section 4.3 of the paper. [cite: 393]
    The matrix 'A_global' here corresponds to matrix 'B' in the paper,
    and 'B_global' corresponds to matrix 'C'.
    """
    NT = element.shape[0]
    A_global = sp.lil_matrix((noedges, noedges))
    B_global = sp.lil_matrix((noedges, NT))

    for j in range(NT):
        elem_nodes = element[j, :]
        coords = coordinate[elem_nodes, :]
        
        # Get global edge numbers for the current element's 3 edges
        I = np.array([
            nodes2edge[elem_nodes[0], elem_nodes[1]] - 1,
            nodes2edge[elem_nodes[1], elem_nodes[2]] - 1,
            nodes2edge[elem_nodes[2], elem_nodes[0]] - 1
        ])
        
        # Determine signs based on normal vector orientation, as per Section 4.3 [cite: 396]
        signum = np.ones(3)
        for k in range(3):
            # If the current element 'j+1' is T- (in the 4th column), the sign is -1.
            if edge2element[I[k], 3] == j + 1:
                signum[k] = -1

        B_local = stimaB(coords)
        A_global[np.ix_(I, I)] += np.diag(signum) @ B_local @ np.diag(signum)
        
        n_vecs = coords[[2, 0, 1], :] - coords[[1, 2, 0], :]
        edge_lengths = np.linalg.norm(n_vecs, axis=1)
        B_global[I, j] = signum * edge_lengths

    return A_global.tocsr(), B_global.tocsr()

# ==============================================================================
# 3. POST-PROCESSING AND VISUALIZATION (from postproce.py)
# These functions are for visualizing the computed displacement and flux.
# ==============================================================================

def show_displacement(elements, coordinates, u_h):
    """
    Visualizes the piecewise constant displacement field u_h.
    This corrected version plots each element individually with its constant value.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the minimum and maximum displacement values to set a consistent color map
    vmin = np.min(u_h)
    vmax = np.max(u_h)
    
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        # Assign the j-th element's displacement value to all 3 of its vertices
        # This creates a flat triangle at the correct height.
        z_values = np.full(3, u_h[j])
        ax.plot_trisurf(verts[:, 0], verts[:, 1], z_values, 
                        vmin=vmin, vmax=vmax, cmap='viridis', edgecolor='k', linewidth=0.2)

    ax.set_title('Displacement u_h')
    ax.view_init(30, -120)
    plt.show()

def flux_eb_eval(coordinate, element, noedges, nodes2edge, edge2element, p_sol):
    """
    Computes the flux vector p_h inside each element from the solution vector.
    This is required for visualization and error estimation.
    This implements part of the logic from Section 7 for post-processing. [cite: 650]
    """
    NT = element.shape[0]
    p_h = np.zeros((3 * NT, 2))

    for j in range(NT):
        elem_nodes = element[j, :]
        coords = coordinate[elem_nodes, :]
        
        I = np.array([
            nodes2edge[elem_nodes[0], elem_nodes[1]] - 1,
            nodes2edge[elem_nodes[1], elem_nodes[2]] - 1,
            nodes2edge[elem_nodes[2], elem_nodes[0]] - 1
        ])

        signum = np.ones(3)
        for k in range(3):
            if edge2element[I[k], 3] == j + 1:
                signum[k] = -1

        area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))
        
        # Basis function evaluation: psi_E = sign * |E| / (2*|T|) * (x - P_opposite)
        # We evaluate p_h = sum(p_sol_i * psi_i) at the element centroid
        centroid = np.mean(coords, axis=0)
        flux_vec = np.zeros(2)
        for i in range(3):
            edge_len = np.linalg.norm(coords[(i+1)%3] - coords[(i+2)%3])
            opposite_vertex = coords[i]
            psi = signum[i] * edge_len / (2 * area) * (centroid - opposite_vertex)
            flux_vec += p_sol[I[i]] * psi
        
        # For visualization, we just assign the constant vector to all 3 vertices
        p_h[3*j : 3*j+3, :] = np.tile(flux_vec, (3, 1))

    return p_h

def show_flux(elements, coordinates, p_h):
    """
    Visualizes the x and y components of the piecewise-constant flux vector p_h.
    This corrected version plots each element individually.
    """
    # Extract one constant (px, py) value for each element from the input vector
    px_elemental = p_h[2::3, 0]
    py_elemental = p_h[2::3, 1]

    fig = plt.figure(figsize=(12, 6))
    
    # --- Plot p_x ---
    ax1 = fig.add_subplot(121, projection='3d')
    # Set color map limits for consistent coloring
    vmin_x, vmax_x = np.min(px_elemental), np.max(px_elemental)
    
    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        # Assign the j-th element's flux value to all 3 of its vertices
        z_values = np.full(3, px_elemental[j])
        ax1.plot_trisurf(verts[:, 0], verts[:, 1], z_values,
                         vmin=vmin_x, vmax=vmax_x, cmap='coolwarm', edgecolor='k', linewidth=0.2)

    ax1.set_title('Flux Component p_x')
    ax1.view_init(30, -120)

    # --- Plot p_y ---
    ax2 = fig.add_subplot(122, projection='3d')
    # Set color map limits for consistent coloring
    vmin_y, vmax_y = np.min(py_elemental), np.max(py_elemental)

    for j, elem in enumerate(elements):
        verts = coordinates[elem]
        # Assign the j-th element's flux value to all 3 of its vertices
        z_values = np.full(3, py_elemental[j])
        ax2.plot_trisurf(verts[:, 0], verts[:, 1], z_values,
                         vmin=vmin_y, vmax=vmax_y, cmap='coolwarm', edgecolor='k', linewidth=0.2)
        
    ax2.set_title('Flux Component p_y')
    ax2.view_init(30, -120)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. Load Mesh and Build Geometric Data ---
    print("Loading mesh files...")
    coordinate, element, dirichlet, neumann = load_mesh_files()
    nodes2element, nodes2edge, noedges, edge2element = edge(coordinate, element)
    print(f"Mesh loaded: {coordinate.shape[0]} nodes, {element.shape[0]} elements, {noedges} edges.")

    # --- 2. Define the Problem (L-shaped domain, Section 9.1) ---
    def f_source(x): # Source term f
        return 0.0 # 

    def u_D_func(x): # Dirichlet boundary value u_D
        return 0.0 # 
        
    def g_func(x, n): # Neumann boundary value g
        # Convert cartesian to polar coordinates [cite: 774]
        r = np.sqrt(x[0]**2 + x[1]**2)
        phi = np.arctan2(x[1], x[0])
        if phi < 0: phi += 2 * np.pi
        
        # Formula from paper [cite: 773]
        val_vec = (2/3) * r**(-1/3) * np.array([-np.sin(phi/3), np.cos(phi/3)])
        return np.dot(val_vec, n)

    # --- 3. Assemble Global System ---
    print("Assembling global system matrices...")
    A, C_T = assemble_global_matrices(coordinate, element, noedges, nodes2edge, edge2element)
    C = C_T.T

    # --- 4. Assemble Right-Hand Side (RHS) Vector ---
    NT = element.shape[0]
    b = np.zeros(noedges + NT)

    # Volume forces (b_f part) [cite: 492]
    b_f = np.zeros(NT)
    for j in range(NT):
        coords = coordinate[element[j, :], :]
        centroid = np.mean(coords, axis=0)
        area = 0.5 * np.linalg.det(np.vstack([np.ones(3), coords.T]))
        b_f[j] = -area * f_source(centroid)
    
    b[noedges:] = b_f

    # --- 5. Incorporate Boundary Conditions ---
    print("Applying boundary conditions...")
    
    # Create a vector 'x' for known boundary values
    x_bc = np.zeros(noedges + NT)
    
    # Get lists of edge indices
    dirichlet_edges_idx = [nodes2edge[d[0], d[1]] - 1 for d in dirichlet]
    neumann_edges_idx = [nodes2edge[n[0], n[1]] - 1 for n in neumann]
    all_bc_edges = set(dirichlet_edges_idx + neumann_edges_idx)
    free_edges_idx = [i for i in range(noedges) if i not in all_bc_edges]

    # Apply Dirichlet conditions to RHS (b_D part) [cite: 509]
    for k, d_edge in enumerate(dirichlet):
        edge_idx = dirichlet_edges_idx[k]
        midpoint = np.mean(coordinate[d_edge, :], axis=0)
        edge_len = np.linalg.norm(coordinate[d_edge[0], :] - coordinate[d_edge[1], :])
        b[edge_idx] += u_D_func(midpoint) * edge_len
        # In this example, u_D is 0, so this adds nothing.

    # Set known Neumann values in the 'x' vector [cite: 534]
    for k, n_edge in enumerate(neumann):
        edge_idx = neumann_edges_idx[k]
        midpoint = np.mean(coordinate[n_edge, :], axis=0)
        vec = coordinate[n_edge[1], :] - coordinate[n_edge[0], :]
        normal = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) # Outward normal
        
        # Ensure normal points outward from the element
        elem_idx_T_plus = edge2element[edge_idx, 2] - 1
        v1, v2 = coordinate[n_edge[0]], coordinate[n_edge[1]]
        v3 = [p for p in element[elem_idx_T_plus] if p not in n_edge][0]
        v3 = coordinate[v3]
        if np.dot(normal, v3 - v1) > 0:
            normal = -normal

        x_bc[edge_idx] = g_func(midpoint, normal)
    
    # Build full system matrix and modify RHS
    # System: [A C_T; C_T.T 0] as per paper's formulation
    SystemMatrix = sp.vstack([
        sp.hstack([A, C_T]),
        sp.hstack([C_T.T, sp.csr_matrix((NT, NT))])
    ])
    
    b -= SystemMatrix @ x_bc
    
# --- 6. Solve the Linear System ---
    print("Solving the linear system...")
    
    # Define free Degrees of Freedom (DoFs)
    # This includes interior edges and ALL BUT ONE of the displacement DoFs
    # to fix the system's singularity.
    free_dofs = free_edges_idx + list(range(noedges, noedges + NT - 1)) #<-- FIX IS HERE

    # Extract the sub-system for free DoFs
    SystemMatrix_csr = SystemMatrix.tocsr()
    A_free = SystemMatrix_csr[np.ix_(free_dofs, free_dofs)]
    b_free = b[free_dofs]
    
    # Solve
    sol_free = spsolve(A_free, b_free)
    
    # Populate the full solution vector
    solution = np.copy(x_bc)
    solution[free_dofs] = sol_free
    
    p_sol = solution[:noedges]  # Solution for flux (p_h)
    u_sol = solution[noedges:]  # Solution for displacement (u_h)
    
    # The last displacement value was fixed to 0.
    u_sol[-1] = 0 #<-- ENSURE THE FIXED VALUE IS SET
    
    print("System solved.")

    # --- 7. Post-Process and Visualize ---
    print("Visualizing results...")
    
    # Calculate flux vectors inside elements for plotting
    p_h_vectors = flux_eb_eval(coordinate, element, noedges, nodes2edge, edge2element, p_sol)
    
    show_displacement(element, coordinate, u_sol)
    show_flux(element, coordinate, p_h_vectors)