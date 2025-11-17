#!/usr/bin/env python3

"""
(C) 2025 Dylan Taylor

Genotype anonymization and transformation functions.
"""

import numpy as np

def random_orthonormal_matrix(N: int, K: int, seed: int = None) -> np.ndarray:

    if K > N:
        raise ValueError("K must be less than or equal to N to form an orthogonal matrix.")
    
    # Generate a random matrix
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((N, K), dtype=np.float32)

    # Orthogonalize columns with QR factorization
    Q, _  = np.linalg.qr(R)

    return Q


def apply_random_projection(G: np.ndarray, Q: np.ndarray) -> np.ndarray:
    
    if G.shape[1] != Q.shape[0]:
        raise ValueError("Number of columns in G must match number of rows in Q.")
    
    # Center rows of G
    G_cent = G - G.mean(axis=1, keepdims=True, dtype=np.float32)

    # Apply projection
    P = G_cent @ Q

    return P


def scale_to_dosage(P) -> np.ndarray:

    row_min = P.min(axis=1, keepdims=True)
    row_max = P.max(axis=1, keepdims=True)
    
    # Avoid division by zero for rows with constant values
    scale = np.where(row_max != row_min, row_max - row_min, 1)
    
    D = 2* (P - row_min) / scale  # Scale to [0,2] row-wise

    return D
