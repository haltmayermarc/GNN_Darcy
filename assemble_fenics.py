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

# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--type", type=str, choices=['pwc','cont'])
args = parser.parse_args()
gparams = args.__dict__

TYPE = gparams["type"]

epsi = 0.01
kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 16.0, 20.0])


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

def create_data(num_input ,num_xy):
    mesh = UnitSquareMesh(num_xy, num_xy)
    V = FunctionSpace(mesh, 'P', 2)
    
    # Define the BC
    u_D = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)

    dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
    num_dofs = V.dim()
    
    dofmap = V.dofmap()
    adj_sets = [set() for _ in range(num_dofs)]

    for cell in cells(mesh):
        cell_dofs = dofmap.cell_dofs(cell.index())
        for i in cell_dofs:
            for j in cell_dofs:
                if i != j:
                    adj_sets[i].add(j)

    # Convert adjacency sets to edge_index
    edges = []
    for i, neighbors in enumerate(adj_sets):
        for j in neighbors:
            edges.append([i, j])
            
    # DOF graph connectivity
    #edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Graph coordinates
    #pos = torch.tensor(dof_coords, dtype=torch.float)

    ne=mesh.cells().shape[0]
    
    pos = dof_coords
    ng=pos.shape[0]

    print("Num of Elements : {}, Num of points : {}".format(ne, ng))
    
    # GRF parameters
    sigma = 1.0     # variance
    ell = 0.3       # correlation length

    # Pairwise distances
    D = cdist(dof_coords, dof_coords)

    # Covariance matrix for GRF
    C = sigma**2 * np.exp(-D**2/(2*ell**2))

    # Generate training and validation data
    train_coeffs_a = []
    train_values_a = []
    train_matrices = []
    train_load_vectors = []
    train_fenics_u = []
    
    validate_coeffs_a = []
    validate_values_a = []
    validate_matrices = []
    validate_load_vectors = []
    validate_fenics_u = []
    

    # TRAINING SET
    np.random.seed(5)
    for _ in tqdm(range(num_input[0])):
        a_sample = np.random.multivariate_normal(
            mean=np.ones(V.dim()),   # mean permeability = 1
            cov=C
        )
        a_sample = np.exp(a_sample)
        a = Function(V)
        a.vector()[:] = a_sample
        
        nx, ny = 64, 64
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)
    
        Z = np.zeros((ny, nx))
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                try:
                    Z[j, i] = a(Point(x, y))
                except RuntimeError:  # if point is outside domain
                    Z[j, i] = np.nan
        if TYPE == "pwc":
            Z_sharp, _ = quantize_field(Z, kappa_values.shape[0], kappa_values)
            a_sample = assign_kappa_quantiles(a_sample, kappa_values)
        else:
            Z_sharp = Z

        train_values_a.append(Z_sharp)
            
        
        #  Define the variational problem
        u = TrialFunction(V)
        v = TestFunction(V)

        f = Constant(1.0)

        a_form = a * dot(grad(u), grad(v)) * dx
        L_form = f * v * dx

        # Assemble matrix
        A = assemble(a_form)
        b = assemble(L_form)
        bc.apply(A, b)
        
        # Solve system in FunctionSpace W
        u_sol = Function(V)
        solve(A, u_sol.vector(), b)
        
        # Store data
        train_coeffs_a.append(a_sample)
        train_matrices.append(A.array())
        train_load_vectors.append(b.get_local())
        train_fenics_u.append(u_sol.vector()[:])

    # VALIDATION SET
    np.random.seed(10)
    for _ in tqdm(range(num_input[1])):
        a_sample = np.random.multivariate_normal(
            mean=np.ones(V.dim()),   # mean permeability = 1
            cov=C
        )
        a_sample = np.exp(a_sample)
        a = Function(V)
        a.vector()[:] = a_sample
        
        nx, ny = 64, 64
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)
    
        Z = np.zeros((ny, nx))
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                try:
                    Z[j, i] = a(Point(x, y))
                except RuntimeError:  # if point is outside domain
                    Z[j, i] = np.nan
        if TYPE == "pwc":
            Z_sharp, _ = quantize_field(Z, kappa_values.shape[0], kappa_values)
            a_sample = assign_kappa_quantiles(a_sample, kappa_values)
        else:
            Z_sharp = Z

        validate_values_a.append(Z_sharp)
        
        
        #  Define the variational problem
        u = TrialFunction(V)
        v = TestFunction(V)

        f = Constant(1.0)

        a_form = a * dot(grad(u), grad(v)) * dx
        L_form = f * v * dx

        # Assemble matrix
        A = assemble(a_form)
        b = assemble(L_form)
        bc.apply(A, b)
        
        # Solve system in FunctionSpace W
        u_sol = Function(V)
        solve(A, u_sol.vector(), b)

        validate_coeffs_a.append(a_sample)
        validate_matrices.append(A.array())
        validate_load_vectors.append(b.get_local())
        validate_fenics_u.append(u_sol.vector()[:])

    return ne, ng, pos, np.array(edges).T, np.array(train_coeffs_a), np.array(train_values_a), np.array(train_matrices), np.array(train_load_vectors), np.array(train_fenics_u),  np.array(validate_coeffs_a), np.array(validate_values_a), np.array(validate_matrices), np.array(validate_load_vectors), np.array(validate_fenics_u)

order='2'
list_num_xy=[10]
num_input=[5000, 1000]
typ='Darcy'

for idx, num in enumerate(list_num_xy):
    ne, ng, p, edges, train_coeffs_a, train_values_a, train_matrices, train_load_vectors, train_fenics_u, validate_coeffs_a, validate_values_a, validate_matrices, validate_load_vectors, validate_fenics_u = create_data(num_input, num)

    # build filename
    base = f"data/P{order}_ne{ne}_{typ}_{num_input[0]}"
    if gparams["type"] is not None:
        mesh_path = f"{base}_{gparams['type']}.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    np.savez(
        mesh_path,
        ne=ne, ng=ng, p=p, edges=edges,
        train_coeffs_a=train_coeffs_a,
        train_values_a=train_values_a,
        train_matrices=train_matrices,
        train_load_vectors = train_load_vectors,
        train_fenics_u=train_fenics_u,
        validate_coeffs_a=validate_coeffs_a,
        validate_values_a=validate_values_a,
        validate_matrices=validate_matrices,
        validate_load_vectors= validate_load_vectors,
        validate_fenics_u=validate_fenics_u
    )
    print(f"Saved data at {mesh_path} for num_xy = {num}")