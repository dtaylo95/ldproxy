#!/usr/bin/env python3

import numpy as np
import pgenlib as pg
import matplotlib.pyplot as plt

#==================#
# Define functions #
#==================#

def pseudo_ROP(G, K, seed=None):
    """
    Generate pseudo-genotypes using random orthonormal projection
    """

    M, N = G.shape
    if K > N:
        raise ValueError("K must be less than the number of columns in G. Cannot produce more pseudo-samples than real samples.")
    
    # Generate Q
    rng = np.random.default_rng(seed=seed)
    Q = rng.normal(size=(N, K))
    Q, _ = np.linalg.qr(Q)

    # Apply projection
    G_cent = G - G.mean(axis=1, keepdims=True)
    P = G_cent @ Q

    return P


def pseudo_Z(G, K, seed=None):
    """
    Generate pseudo-genotypes using Z-scores of random trait regression
    """

    M, N = G.shape
    
    # Generate traits
    rng = np.random.default_rng(seed=seed)
    T = rng.normal(size=(N, K))

    # center (include intercept)
    Gc = G - G.mean(axis=1, keepdims=True)   # (M x N)
    Tc = T - T.mean(axis=0, keepdims=True)   # (N x K)

    # dot products
    XY = Gc @ Tc                             # (M x K)
    GG = np.sum(Gc * Gc, axis=1)             # (M,)
    TT = np.sum(Tc * Tc, axis=0)             # (K,)

    # compute r (M x K)
    # avoid division by zero
    denom = np.sqrt(np.outer(GG, TT))        # (M x K)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = XY / denom

    # clamp r in (-1,1) for numerical stability
    R = np.clip(R, -1 + 1e-15, 1 - 1e-15)

    # convert r -> t with df = n-2
    df = N - 2
    Z = R * np.sqrt(df / (1 - R * R))

    return Z


def cov_to_corr(cov):
    # Standard deviations are the square roots of the diagonal
    std = np.sqrt(np.diag(cov))
    # Outer product of std gives the normalization matrix
    denom = np.outer(std, std)
    # Elementwise division
    corr = cov / denom
    # Numerical cleanup: force diagonal to exactly 1
    np.fill_diagonal(corr, 1.0)
    return corr


#============#
# Set params #
#============#

K = 400


#===========#
# Load data #
#===========#

pgen_file = "/scratch16/rmccoy22/dtaylo95/ldproxy/tests/data/test_input.pgen"

reader = pg.PgenReader(pgen_file.encode('utf-8'))

n_variants = reader.get_variant_ct()
n_samples = reader.get_raw_sample_ct()

# Allocate output matrix: variants × samples
G = np.empty((n_variants, n_samples), dtype=np.int8)

# Temporary buffer for reading one variant at a time
buf = np.empty(n_samples, dtype=np.int8)

# Iterate and read
for vidx in range(n_variants):
    reader.read(vidx, buf, allele_idx=1)
    G[vidx, :] = buf  # copy into matrix


#======================#
# Get pseudo-genotypes #
#======================#

P_ROP = pseudo_ROP(G, K, 42)
P_Z = pseudo_Z(G, K, 42)


#========================#
# Calculate correlations #
#========================#

corr_G = np.corrcoef(G)

cov_ROP = P_ROP @ P_ROP.T
corr_ROP = cov_to_corr(cov_ROP)

corr_ROP2 = np.corrcoef(P_ROP)


triu_ind = np.triu_indices(n_variants, k=1)

triu_G = corr_G[triu_ind]
triu_ROP = corr_ROP[triu_ind]
triu_ROP2 = corr_ROP2[triu_ind]


#==================#
# Plot comparisons #
#==================#

fig, axes = plt.subplots(ncols=3, figsize=(18,6))
axes[0].scatter(x=triu_G, y=triu_ROP, s=1, c='black')
axes[0].axline((0,0), slope=1, ls='--', lw=1, c='red')
axes[0].set_xlabel('SNP correlations')
axes[0].set_ylabel('ROP correlations')
axes[1].scatter(x=triu_G, y=triu_ROP2, s=1, c='black')
axes[1].axline((0,0), slope=1, ls='--', lw=1, c='red')
axes[1].set_xlabel('SNP correlations')
axes[1].set_ylabel('ROP2 correlations')
axes[2].scatter(x=triu_ROP, y=triu_ROP2, s=1, c='black')
axes[2].axline((0,0), slope=1, ls='--', lw=1, c='red')
axes[2].set_xlabel('ROP correlations')
axes[2].set_ylabel('ROP2 correlations')
fig.tight_layout()
fig.savefig('correlation_comparison.png', dpi=300)
