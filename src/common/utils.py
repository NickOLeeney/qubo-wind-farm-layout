import numpy as np
import scipy.sparse as sp

def safe_cast_uint16(x):
    x = np.asarray(x)

    info = np.iinfo(np.uint16)

    if x.min() < info.min or x.max() > info.max:
        raise ValueError(
            f"Valori fuori range uint16: "
            f"[{x.min()}, {x.max()}], "
            f"range ammesso [{info.min}, {info.max}]"
        )

    ram_before = x.nbytes

    x_uint16 = np.rint(x).astype(np.uint16)

    ram_after = x_uint16.nbytes
    ram_saved = ram_before - ram_after
    saved_pct = 100 * ram_saved / ram_before

    print(f"RAM before : {ram_before / 1024**2:.2f} MB")
    print(f"RAM after  : {ram_after / 1024**2:.2f} MB")
    print(f"RAM saved  : {ram_saved / 1024**2:.2f} MB ({saved_pct:.1f}%)")

    return x_uint16


def print_ram():
    import os
    import psutil

    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())

    print(f"System available : {mem.available / 1024**3:.2f} GB")
    print(f"System used      : {mem.percent:.1f}%")
    print(f"Python process   : {process.memory_info().rss / 1024**3:.2f} GB")


def sparsify_wake_by_percentile(
    wake_loss_matrix,
    percentile,
    n_turbines=81,
):  
    if not percentile:
        return None
    
    L = np.asarray(wake_loss_matrix, dtype=float)

    # Consideriamo ogni interazione una sola volta
    triu_idx = np.triu_indices_from(L, k=1)
    losses = L[triu_idx]

    # Threshold corrispondente al percentile scelto
    threshold = np.percentile(losses, percentile)

    # Drop delle interazioni deboli
    L_sparse = L.copy()
    L_sparse[L_sparse < threshold] = 0.0

    np.fill_diagonal(L_sparse, 0.0)

    # Statistiche
    total_pairs = len(losses)

    retained_pairs = np.count_nonzero(
        L_sparse[triu_idx]
    )

    density = retained_pairs / total_pairs

    # Worst-case error bound per un layout di K turbine
    max_active_pairs = (
        n_turbines * (n_turbines - 1) // 2
    )

    max_error_bound = (
        max_active_pairs * threshold
    )

    L_sparse = sparsify_wake_by_percentile(L_sparse)

    return L_sparse, {
        "percentile": percentile,
        "threshold": threshold,
        "retained_pairs": retained_pairs,
        "total_pairs": total_pairs,
        "density": density,
        "max_error_bound": max_error_bound,
    }


def upper_triangular_csr(L):
    n = L.shape[0]

    counts = np.array([
        np.count_nonzero(L[i, i + 1:])
        for i in range(n)
    ], dtype=np.int64)

    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    nnz = int(indptr[-1])

    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=L.dtype)

    pos = 0

    for i in range(n):
        row = L[i, i + 1:]
        nz = np.flatnonzero(row)

        k = len(nz)

        indices[pos:pos + k] = i + 1 + nz
        data[pos:pos + k] = row[nz]

        pos += k

    return sp.csr_matrix(
        (data, indices, indptr),
        shape=(n, n),
    )