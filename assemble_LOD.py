import numpy as np
import pandas as pd
import scipy
from scipy import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from datetime import datetime
import os
import random
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dolfin import *
from mshr import *
from scipy.spatial.distance import cdist
from LOD import *

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--type", type=str, choices=['quantile','checkerboard', 'horizontal', 'vertical'])
parser.add_argument("--H", type=int, default=3)
parser.add_argument("--h", type=int, default=7)
parser.add_argument("--k", type=int, default=3)
args = parser.parse_args()
gparams = args.__dict__

TYPE = gparams["type"]
H = 2**(-gparams["H"])
h = 2**(-gparams["h"])
k = gparams["k"]

#epsi = 0.01
#kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 16.0, 20.0])

def compute_kappa_per_element(coarse_elems, kappa_node):
    """
    Compute elementwise-constant kappa on coarse mesh.

    Returns
    -------
    kappa_elem : (N_elem,) ndarray
    """
    kappa_elem = np.zeros(len(coarse_elems))
    for l, elem in enumerate(coarse_elems):
        kappa_elem[l] = np.mean(kappa_node[list(elem)])
    return kappa_elem

def make_fast_kappa_uniform(h, Nx, kappa_elem):
    def kappa(x, y):
        i = min(int(x // h), Nx - 1)
        j = min(int(y // h), Nx - 1)

        # quad index
        q = j * Nx + i

        # two triangles per quad
        # consistent with quads_to_tris(bl-tr)
        if (x - i*h) + (y - j*h) <= h:
            l = 2*q       # lower-left triangle
        else:
            l = 2*q + 1   # upper-right triangle

        return kappa_elem[l]

    return kappa

"""
def f_const(x, y):
    return 1.0

def assign_kappa_quantiles(a_sample, kappa_values):
    K = len(kappa_values)
    edges = np.quantile(a_sample, np.linspace(0, 1, K + 1))
    bins = np.digitize(a_sample, edges[1:-1], right=True)
    bins = np.clip(bins, 0, K - 1)
    
    return kappa_values[bins]

def quantize_field(Z, n, kappa_values=None):
    flat = Z.flatten()
    # Compute quantile boundaries (n+1 edges)
    q_edges = np.quantile(flat, np.linspace(0, 1, n+1))

    if kappa_values is None:
        kappa_values = np.arange(1, n+1)
    kappa_values = np.asarray(kappa_values)

    # Digitize field values based on quantile bins
    bin_indices = np.digitize(flat, q_edges[1:-1], right=False)
    Z_sharp = kappa_values[bin_indices].reshape(Z.shape)

    return Z_sharp, q_edges
"""

def make_LOD_data_quantile(Nx, Ny, refine, grid, adjacency, fine_in_coarse, kappa, B_H, C_h, f_h, P_h):
    coarse_nodes = grid["coarse_nodes"]
    fine_nodes   = grid["fine_nodes"]
    fine_elems   = grid["fine_elems"]
    
    # g_values = g.vector().get_local()
    # q_edges = np.quantile(g_values, np.linspace(0, 1, kappa_values.shape[0]+1))
    
    #def quantize_scalar(value, q_edges, kappa_values):
    #    # Return the kappa level corresponding to value
    #    idx = np.searchsorted(q_edges[1:], value, side="right")
    #    return kappa_values[min(idx, len(kappa_values)-1)]
    
    #def kappa_sharp(g, q_edges, kappa_values, x, y):
    #    val = g(Point(x, y))          
    #    return quantize_scalar(val, q_edges, kappa_values)

    
    A_dc, M_dc, sigma = build_fine_element_matrices(grid, lambda x, y: kappa(x,y))
    Nh = fine_nodes.shape[0]
    A_h, M_h = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)
    
    
    # Dirichlet boundary on fine grid
    #bdry = boundary_mask_fine(fine_nodes)
    #free_mask = ~bdry
    #free_idx = np.where(free_mask)[0]
    #A_free = A_h[np.ix_(free_idx, free_idx)]
    #f_free = f_full[free_idx]
    
    
    # compute global correctors Q_h 
    Q_h = computeCorrections(grid, k, adjacency, fine_in_coarse, A_h, B_H, C_h, kappa, n_jobs=-1)
    
    V_ms = P_h + Q_h  # (N_H x N_h)

    A_lod = V_ms @ A_h @ V_ms.T   # (N_H x N_H)
    f = V_ms @ f_h            # (N_H,)

    interior = np.where(np.diag(B_H) > 0.5)[0]  # interior coarse node indices
    A_lod = A_lod[np.ix_(interior, interior)]
    f_lod  = f[interior]
    

    # coarse interior mask (Dirichlet on ∂Ω in coarse space)
    #coarse_interior_mask = np.ones(N_H, dtype=bool)
    #for j in range(Ny+1):
    #    for i in range(Nx+1):
    #        idx = j*(Nx+1) + i
    #        if i==0 or i==Nx or j==0 or j==Ny:
    #            coarse_interior_mask[idx] = False
                
    #A_lod, f_lod = get_LOD_matrix_rhs(A_free, f_free, P_free, Q_free, coarse_interior_mask)
    
    return A_lod, f_lod, #A_h, M_h, Q_h

"""
def make_LOD_data(Nx, Ny, refine, grid, g):
    coarse_nodes = grid["coarse_nodes"]
    coarse_elems = grid["coarse_elems"]
    fine_nodes   = grid["fine_nodes"]
    fine_elems   = grid["fine_elems"]
    
    A_dc, M_dc, sigma = build_fine_element_matrices_var_kappa_3x3(grid, g)
    Nh = fine_nodes.shape[0]
    A_h, M_h = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)
    
    # RHS
    f_full = assemble_load_quad(fine_nodes, fine_elems, f_const)

    # Dirichlet boundary on fine grid
    bdry = boundary_mask_fine(fine_nodes)
    free_mask = ~bdry
    free_idx = np.where(free_mask)[0]
    A_free = A_h[np.ix_(free_idx, free_idx)]
    f_free = f_full[free_idx]

    # interpolation P and coarse boundary mask
    P = build_P_quad_unique(Nx, Ny, refine)       
    P_h = P.T                                     
    P_free = P[free_idx, :]                      
    N_H = P.shape[1]
    B_H = build_B_H(coarse_nodes, Nx, Ny)

    # build patches per coarse element (no oversampling)
    R_h_list, R_H_list, T_H_list, fine_elems_in_coarse = build_patch_mappings(grid)
    NTH = len(T_H_list)
    
    # compute global correctors Q_h (Algorithm 1)
    Q_h = computeCorrections_algorithm(
        Nh, N_H, NTH,
        A_dc, M_dc, sigma,
        B_H, P_h,
        R_h_list, R_H_list, T_H_list, fine_elems_in_coarse
    )

    # restrict Q_h to free DOFs
    Q_free = Q_h[:, free_idx]   # (N_H x N_free)

    # coarse interior mask (Dirichlet on ∂Ω in coarse space)
    coarse_interior_mask = np.ones(N_H, dtype=bool)
    for j in range(Ny+1):
        for i in range(Nx+1):
            idx = j*(Nx+1) + i
            if i==0 or i==Nx or j==0 or j==Ny:
                coarse_interior_mask[idx] = False
                
    A_lod, f_lod = get_LOD_matrix_rhs(A_free, f_free, P_free, Q_free, coarse_interior_mask)
    
    return A_lod, f_lod
"""


class SEGaussianSamplerSVD:
    
    def __init__(self, pos, sigma=1.0, ell=0.3, mean=1.0, tol=1e-8):
        self.pos = np.asarray(pos, dtype=float)
        self.V = self.pos.shape[0]

        # Pairwise distances
        D = cdist(self.pos, self.pos)

        # Covariance matrix for GRF
        C = (sigma**2) * np.exp(-(D**2) / (2 * ell**2))

        U, s, Vt = np.linalg.svd(C, full_matrices=False)

        if np.min(s) < -tol:
            raise ValueError(f"Covariance not PSD: min eigen/singular {np.min(s)} < -tol")
        s = np.clip(s, 0.0, None)

        self.A = U * np.sqrt(s)[None, :]
        self.mean = mean

    def sample(self, rng: np.random.Generator):
        z = rng.normal(0.0, 1.0, size=(self.V,))
        return self.mean + self.A @ z

def create_dataset(num_input, H, h, kappa_values):
    # Create coarse and fine mesh
    Nx = int(1 / H)
    Ny = Nx
    refine = int(H / h)

    mesh_data = build_triangular_mesh(Nx, Ny, refine)
    
    coarse_nodes = mesh_data["coarse_nodes"]
    coarse_elems = mesh_data["coarse_elems"]
    fine_nodes   = mesh_data["fine_nodes"]
    fine_elems   = mesh_data["fine_elems"]
    
    N_H = coarse_nodes.shape[0]
    N_h = fine_nodes.shape[0]
    
    V_dim = coarse_nodes.shape[0]
    
    # Mesh connectivity and adjacency
    coarse_row, coarse_col = elems_to_coo(coarse_elems)
    edges = np.vstack([coarse_row, coarse_col])
    
    adjacency = build_coarse_adjacency_edge(coarse_elems)
    fine_in_coarse = precompute_fine_in_coarse(mesh_data)
    pos = fine_nodes
    print("Meshing finished...")
    
    # GRF parameters
    sigma = 1.0     # variance
    ell = 0.3       # correlation length
    
    # Construct SE Kernel sampler for GRF
    if TYPE == "quantile":
        sampler = SEGaussianSamplerSVD(pos, sigma=sigma, ell=ell, mean=1.0, tol=1e-8)
    
    # Function space for GRF sampling
    mesh_fenics = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh_fenics, "CG", 1)
    K = np.zeros([Nx, Ny])
    
    ###########################################################
    # Generate all data that do not depend 
    # on a given coefficient instance or patch
    ###########################################################
    
    # Interpolation matrix P_h from the 2019 paper
    print("Building P_h...")
    P_h = build_P_triangular(mesh_data)
    # Boundary matrix
    print("Building B_H...")
    B_H = build_B_H(coarse_nodes, Nx, Ny)
    interior = np.where(np.diag(B_H) > 0.5)[0]
    # Fine forcing term rhs vector
    print("Building f_h...")
    f_h = assemble_load_tri(
        fine_nodes, fine_elems, lambda x, y: 1.0
    )
    # From the 2020 SIAM book
    print("Building C_h...")
    C_h = build_IH_quasi_interpolation(mesh_data)
    
    # Generate training and validation data
    train_coeffs_a = []
    train_matrices = []
    train_load_vectors = []
    train_fenics_u = []
    
    validate_coeffs_a = []
    validate_matrices = []
    validate_load_vectors = []
    validate_fenics_u = []
    
    
    # TRAINING SET
    np.random.seed(5)
    for _ in tqdm(range(num_input[0])):
        if TYPE == "quantile":
            rng = np.random.default_rng()
            a_sample = sampler.sample(rng)
            a_sample = np.exp(a_sample)
            #a = Function(V)
            #a.vector()[:] = a_sample
            q_edges = np.quantile(a_sample, np.linspace(0, 1, kappa_values.shape[0]+1))
            
            kappa_shuffled = np.asarray(kappa_values, dtype=float).copy()
            rng.shuffle(kappa_shuffled)
            
            bins = np.searchsorted(q_edges[1:], a_sample, side="right")
            bins = np.clip(bins, 0, len(kappa_values)-1)
            kappa_node = kappa_shuffled[bins]
            
            kappa_elem = compute_kappa_per_element(fine_elems, kappa_node)
            kappa = make_fast_kappa_uniform(h, int(1 / h), kappa_elem)
            
            train_coeffs_a.append(kappa_node)

            #A_LOD_matrix, f_LOD_vector = make_LOD_data_quantile(Nx, Ny, refine, mesh_data, adjacency, fine_in_coarse, kappa, B_H, C_h, f_h, P_h)
            #train_matrices.append(A_LOD_matrix)
            #train_load_vectors.append(f_LOD_vector)
            A_LOD_matrix, f_LOD_vector = make_LOD_data_quantile(
                    Nx, Ny, refine, mesh_data,
                    adjacency, fine_in_coarse,
                    kappa, B_H, C_h, f_h, P_h
            )

            train_matrices.append(A_LOD_matrix)
            train_load_vectors.append(f_LOD_vector)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
            
            #u_H = np.zeros(A_LOD_matrix.shape[0])
            #u_H[interior] = u_lod
            train_fenics_u.append(u_lod)
            
        elif TYPE in ["checkerboard", "horizontal", "vertical"]:
            rng = np.random.default_rng()

            # --- checkerboard ---
            if TYPE == "checkerboard":
                for j in range(Ny):
                    for i in range(Nx):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]

            # --- horizontal stripes ---
            elif TYPE == "horizontal":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for j in range(Nx):
                    stripe_id = min(int(j / Ny * n_stripes), n_stripes - 1)
                    K[j, :] = stripe_vals[stripe_id]

            # --- vertical stripes ---
            elif TYPE == "vertical":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for i in range(Ny):
                    stripe_id = min(int(i / Nx * n_stripes), n_stripes - 1)
                    K[:, i] = stripe_vals[stripe_id]

            # --- callable coefficient ---
            def a(x, y):
                x = min(max(x, 0.0), 1.0 - 1e-14)
                y = min(max(y, 0.0), 1.0 - 1e-14)

                i = int(x * Ny)
                j = int(y * Nx)

                return K[j, i]

            # --- sample on coarse nodes ---
            a_sample = np.array([a(p[0], p[1]) for p in pos])
            train_coeffs_a.append(a_sample)

            A_LOD_matrix, f_LOD_vector = make_LOD_data(
                num_xy, num_xy, refine, mesh_data, a
            )

            train_matrices.append(A_LOD_matrix)
            train_load_vectors.append(f_LOD_vector)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(
                    A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]),
                    f_LOD_vector,
                    rcond=None
                )[0]
            train_fenics_u.append(u_lod)
            
    # VALIDATION SET
    np.random.seed(10)
    for _ in tqdm(range(num_input[1])):
        if TYPE == "quantile":
            rng = np.random.default_rng()
            a_sample = sampler.sample(rng)
            a_sample = np.exp(a_sample)
            #a = Function(V)
            #a.vector()[:] = a_sample
            q_edges = np.quantile(a_sample, np.linspace(0, 1, kappa_values.shape[0]+1))
            
            kappa_shuffled = np.asarray(kappa_values, dtype=float).copy()
            rng.shuffle(kappa_shuffled)
            
            bins = np.searchsorted(q_edges[1:], a_sample, side="right")
            bins = np.clip(bins, 0, len(kappa_values)-1)
            kappa_node = kappa_shuffled[bins]
            
            kappa_elem = compute_kappa_per_element(fine_elems, kappa_node)
            kappa = make_fast_kappa_uniform(h, int(1 / h), kappa_elem)
            
            train_coeffs_a.append(kappa_node)

            #A_LOD_matrix, f_LOD_vector = make_LOD_data_quantile(Nx, Ny, refine, mesh_data, adjacency, fine_in_coarse, kappa, B_H, C_h, f_h, P_h)
            #validate_matrices.append(A_LOD_matrix)
            #validate_load_vectors.append(f_LOD_vector)
            A_LOD_matrix, f_LOD_vector = make_LOD_data_quantile(
                Nx, Ny, refine, mesh_data,
                adjacency, fine_in_coarse,
                kappa, B_H, C_h, f_h, P_h
            )

            validate_matrices.append(A_LOD_matrix)
            validate_load_vectors.append(f_LOD_vector)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
                
            #u_H = np.zeros(A_LOD_matrix.shape[0])
            #u_H[interior] = u_lod
            validate_fenics_u.append(u_lod)
            
        elif TYPE in ["checkerboard", "horizontal_stripes", "vertical_stripes"]:
            rng = np.random.default_rng()

            # --- checkerboard ---
            if TYPE == "checkerboard":
                for j in range(num_xy):
                    for i in range(num_xy):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]

            # --- horizontal stripes ---
            elif TYPE == "horizontal_stripes":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for j in range(num_xy):
                    stripe_id = min(int(j / num_xy * n_stripes), n_stripes - 1)
                    K[j, :] = stripe_vals[stripe_id]

            # --- vertical stripes ---
            elif TYPE == "vertical_stripes":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for i in range(num_xy):
                    stripe_id = min(int(i / num_xy * n_stripes), n_stripes - 1)
                    K[:, i] = stripe_vals[stripe_id]

            # --- callable coefficient ---
            def a(x, y):
                x = min(max(x, 0.0), 1.0 - 1e-14)
                y = min(max(y, 0.0), 1.0 - 1e-14)

                i = int(x * num_xy)
                j = int(y * num_xy)

                return K[j, i]

            # --- sample on coarse nodes ---
            a_sample = np.array([a(p[0], p[1]) for p in pos])
            validate_coeffs_a.append(a_sample)

            A_LOD_matrix, f_LOD_vector = make_LOD_data(
                num_xy, num_xy, refine, mesh_data, a
            )

            validate_matrices.append(A_LOD_matrix)
            validate_load_vectors.append(f_LOD_vector)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(
                    A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]),
                    f_LOD_vector,
                    rcond=None
                )[0]
            validate_fenics_u.append(u_lod)
    
    #return pos, edges, np.array(train_coeffs_a), np.array(train_matrices), np.array(train_load_vectors), np.array(train_fenics_u),  np.array(validate_coeffs_a), np.array(validate_matrices), np.array(validate_load_vectors), np.array(validate_fenics_u)
    return (
    pos, edges,
    np.array(train_coeffs_a),
    np.array(train_matrices),
    np.array(train_load_vectors),
    np.array(train_fenics_u),
    np.array(validate_coeffs_a),
    np.array(validate_matrices),
    np.array(validate_load_vectors),
    np.array(validate_fenics_u),
    coarse_nodes,
    coarse_elems,
    fine_nodes,
    fine_elems
)


order='1'
list_num_xy=[63]
num_input=[5000, 500]
typ='Darcy'
#refine = 2

epsi = 0.01
kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 16.0, 20.0])

for idx, num in enumerate(list_num_xy):
    #p, edges, train_coeffs_a, train_matrices, train_load_vectors, train_fenics_u, validate_coeffs_a, validate_matrices, validate_load_vectors, validate_fenics_u = create_dataset(num_input, H, h, kappa_values)
    (
        p, edges,
        train_coeffs_a,
        train_matrices,
        train_load_vectors,
        train_fenics_u,
        validate_coeffs_a,
        validate_matrices,
        validate_load_vectors,
        validate_fenics_u,
        coarse_nodes,
        coarse_elems,
        fine_nodes,
        fine_elems
    ) = create_dataset(num_input, H, h, kappa_values)


    # build filename
    base = f"data/P{order}_ne{H}_{typ}_{num_input[0]}"
    if gparams["type"] is not None:
        mesh_path = f"{base}_{gparams['type']}.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    np.savez(
        mesh_path,
        p=p,
        edges=edges,

        coarse_nodes=coarse_nodes,
        coarse_elems=coarse_elems,
        fine_nodes=fine_nodes,
        fine_elems=fine_elems,
        
        train_coeffs_a=train_coeffs_a,
        train_matrices=train_matrices,
        train_load_vectors=train_load_vectors,
        train_u=train_fenics_u,

        validate_coeffs_a=validate_coeffs_a,
        validate_matrices=validate_matrices,
        validate_load_vectors=validate_load_vectors,
        validate_u=validate_fenics_u
    )

    """
    np.savez(
        mesh_path,
        p=p, edges=edges,
        train_coeffs_a=train_coeffs_a,
        train_matrices=train_matrices,
        train_load_vectors = train_load_vectors,
        train_fenics_u=train_fenics_u,
        validate_coeffs_a=validate_coeffs_a,
        validate_matrices=validate_matrices,
        validate_load_vectors= validate_load_vectors,
        validate_fenics_u=validate_fenics_u
    )
    """
    print(f"Saved data at {mesh_path} for num_xy = {num}")