# -*- coding: utf-8 -*-
"""Non-specific Welch's t-test up to 3rd order (univariate per time sample).
We compare two groups: fixed (id=0) vs random (id=1).
For order m in {1,2,3}, we t-test the transformed variable y = x**m.
Everything is vectorized across 1000 time samples; streaming over traces.
"""
import numpy as np

def update_group_sums(traces_i1: np.ndarray, ids_u1: np.ndarray, sums, orders=(1,2,3)):
    """Update running sums for each group and order with a block of traces.
    traces_i1: (B,P) int8
    ids_u1   : (B,) uint8 where 0=fixed, 1=random
    sums     : dict with keys ['n0','n1','sum0','sum1','sum20','sum21'] for each order m.
               For convenience, sums['sum0'][m] is a (P,) array for order m.
    """
    X = traces_i1.astype(np.int64)  # safe integer arithmetic
    P = X.shape[1]
    # Build masks and groups
    mask0 = (ids_u1 == 0)
    mask1 = ~mask0
    X0 = X[mask0, :]   # (n0,P)
    X1 = X[mask1, :]   # (n1,P)
    n0 = X0.shape[0]; n1 = X1.shape[0]
    if n0:
        sums['n0'] += n0
    if n1:
        sums['n1'] += n1

    for m in orders:
        if n0:
            Y0 = np.power(X0, m, dtype=np.int64)      # (n0,P)
            sums['sum0'][m]  += Y0.sum(axis=0)        # Σ y
            sums['sum20'][m] += (Y0*Y0).sum(axis=0)   # Σ y^2
        if n1:
            Y1 = np.power(X1, m, dtype=np.int64)
            sums['sum1'][m]  += Y1.sum(axis=0)
            sums['sum21'][m] += (Y1*Y1).sum(axis=0)

def welch_t_from_sums(sums, order: int):
    """Compute Welch's t-values (array length P) for the specified order.
    Uses the running sums in 'sums'."""
    n0 = max(0, int(sums['n0']))
    n1 = max(0, int(sums['n1']))
    if n0 < 2 or n1 < 2:
        # Not enough samples to compute variance; return zeros
        return np.zeros_like(sums['sum0'][order], dtype=np.float64)

    s0  = sums['sum0'][order].astype(np.float64)
    s20 = sums['sum20'][order].astype(np.float64)
    s1  = sums['sum1'][order].astype(np.float64)
    s21 = sums['sum21'][order].astype(np.float64)

    mu0 = s0 / n0
    mu1 = s1 / n1
    # sample variances
    var0 = (s20 - (s0*s0)/n0) / (n0 - 1)
    var1 = (s21 - (s1*s1)/n1) / (n1 - 1)

    # avoid negative numerical noise
    var0 = np.maximum(var0, 0.0)
    var1 = np.maximum(var1, 0.0)

    denom = np.sqrt(var0 / n0 + var1 / n1)
    denom = np.where(denom == 0.0, np.inf, denom)  # avoid divide-by-zero
    tvals = (mu0 - mu1) / denom
    return tvals
