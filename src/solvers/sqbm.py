"""
SQBM+ Python Client
Wrapper for the SQBM+ REST APIs (Simulated Quantum-inspired Bifurcation Machine plus).
"""

import io
import time
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SQBMResult:
    id: str
    time: float
    wait: float
    message: str
    runs: int
    value: float
    result: list[int]
    param: dict
    count: int
    others: list[dict] = field(default_factory=list)

    def save(self, path) -> "Path":
        """Serialize to JSON. Creates parent dirs automatically."""
        from pathlib import Path
        import json
        from dataclasses import asdict
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return p

    @classmethod
    def load(cls, path) -> "SQBMResult":
        """Deserialize from a JSON file written by save()."""
        from pathlib import Path
        import json
        with open(Path(path)) as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"SQBMResult(\n"
            f"  value   = {self.value}\n"
            f"  result  = {self.result}\n"
            f"  message = '{self.message}'\n"
            f"  runs    = {self.runs}\n"
            f"  time    = {self.time}s\n"
            f"  param   = {self.param}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SQBMClient:
    """
    Client for the SQBM+ REST APIs.

    Args:
        host:           IP or hostname of the SQBM+ server.
        port:           Server port (default 8000).
        timeout:        HTTP request timeout in seconds (default 120). This is the
                        *transport* timeout (upload + wait for response); it is NOT
                        the solver computation budget (that is the `timeout` query
                        parameter passed to solve_qubo / solve_qubo_file).
        problem_format: Encoding used when a numpy matrix is submitted via
                        solve_qubo(): "matrixmarket" (text, default) or "hdf5"
                        (binary CSR). HDF5 produces a smaller payload and parses
                        faster server-side; it is worth it for dense / large
                        matrices. Aliases: "mm"/"mtx" -> matrixmarket, "h5"/"hdf" -> hdf5.
    """

    _SOLVERS = {"qubo", "qplib", "pubo", "tsp", "shift", "qap"}
    _FORMATS = {"matrixmarket", "hdf5"}

    # Accepted spellings mapped to the canonical format name.
    _FORMAT_ALIASES = {
        "matrixmarket": "matrixmarket", "mm": "matrixmarket", "mtx": "matrixmarket",
        "hdf5": "hdf5", "h5": "hdf5", "hdf": "hdf5",
    }

    def __init__(
        self,
        host: str,
        port: int = 8000,
        timeout: int = 120,
        problem_format: str = "hdf5",
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.problem_format = self._normalize_format(problem_format)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/octet-stream"})

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_format(cls, fmt: str) -> str:
        key = str(fmt).strip().lower()
        if key not in cls._FORMAT_ALIASES:
            raise ValueError(
                f"Unknown problem_format {fmt!r}. "
                f"Use one of: {sorted(cls._FORMATS)} (aliases: mm, mtx, h5, hdf)."
            )
        return cls._FORMAT_ALIASES[key]

    def _encode_matrix(self, Q: np.ndarray, problem_format: Optional[str] = None) -> bytes:
        """Encode Q using the given format, or the instance default if None."""
        fmt = self._normalize_format(problem_format) if problem_format else self.problem_format
        if fmt == "matrixmarket":
            return self.matrix_to_matrixmarket(Q)
        if fmt == "hdf5":
            return self.matrix_to_hdf5(Q)
        raise ValueError(f"Unsupported problem_format: {fmt!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Shared symmetrization (used by BOTH encoders -> identical semantics)
    # ------------------------------------------------------------------

    @staticmethod
    def _symmetrized_lower(Q: np.ndarray) -> np.ndarray:
        """
        Fold Q into the lower triangle:
          - off-diagonal (i > j) -> Q[i, j] + Q[j, i]  (full pairwise coefficient)
          - diagonal            -> Q[i, i]
          - upper triangle      -> 0
        Both the MatrixMarket and the HDF5 encoders build from this same matrix,
        so the two formats represent exactly the same problem.
        """
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a 2D square matrix.")
        L = np.tril(Q + Q.T, k=-1)
        np.fill_diagonal(L, np.diag(Q))
        return L

    # ------------------------------------------------------------------
    # Numpy matrix -> MatrixMarket (coordinate, text) conversion
    # ------------------------------------------------------------------

    @staticmethod
    def matrix_to_matrixmarket(Q: np.ndarray) -> bytes:
        L = SQBMClient._symmetrized_lower(Q)
        n = L.shape[0]

        rows, cols = np.nonzero(L)          # rows >= cols (lower triangle + diagonal)
        vals = L[rows, cols]

        header = f"%%MatrixMarket matrix coordinate real general\n{n} {n} {vals.size}\n"
        buf = io.BytesIO()
        buf.write(header.encode("utf-8"))
        np.savetxt(buf, np.column_stack((rows + 1, cols + 1, vals)),
                   fmt=("%d", "%d", "%.12g"))
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Numpy matrix -> HDF5 (sparse CSR, binary) conversion
    # ------------------------------------------------------------------

    @staticmethod
    def matrix_to_hdf5(Q: np.ndarray) -> bytes:
        """
        Encode Q as an SQBM+-compatible sparse-CSR HDF5 payload.

        Layout mandated by the SQBM+ V2 user manual (section 3.3.1.4.2):

            +-Group("/qubo")
              +-DataSet("/qubo/data")     <- Attribute["format"]="csr" lives HERE
              +-DataSet("/qubo/indptr")
              +-DataSet("/qubo/indices")

        The `format="csr"` attribute must be attached to the /qubo/data *dataset*,
        NOT to the /qubo group (attaching it to the group makes the server reject
        the file with HTTP 400 "unknown format"). Required dtypes are fixed by the
        manual and are not configurable:

            data    -> float32
            indices -> uint32
            indptr  -> uint32

        Only the lower triangle (including the diagonal) is stored: the server's
        objective value is computed from the lower-triangular elements only, so
        _symmetrized_lower folds Q + Q.T into it — identical semantics to the
        MatrixMarket encoder. Verified against the live server: CSR, dense and
        MatrixMarket return the same objective value for the same Q.

        Requires h5py (`pip install h5py`, >= 2.9 for the in-memory file support).
        """
        try:
            import h5py
        except ImportError as e:  # keep h5py optional for MatrixMarket-only users
            raise ImportError(
                "HDF5 encoding requires h5py. Install it with: pip install h5py"
            ) from e

        L = SQBMClient._symmetrized_lower(Q)
        n = L.shape[0]

        # np.nonzero returns row-major order, so columns are already ascending
        # within each row -> a valid CSR ordering.
        rows, cols = np.nonzero(L)
        data = L[rows, cols]
        counts = np.bincount(rows, minlength=n)          # non-zeros per row
        indptr = np.concatenate(([0], np.cumsum(counts)))

        bio = io.BytesIO()
        with h5py.File(bio, "w") as f:
            g = f.create_group("qubo")
            ds = g.create_dataset("data", data=data.astype(np.float32))
            ds.attrs["format"] = "csr"
            g.create_dataset("indptr", data=indptr.astype(np.uint32))
            g.create_dataset("indices", data=cols.astype(np.uint32))
        return bio.getvalue()

    # ------------------------------------------------------------------
    # Internal utility
    # ------------------------------------------------------------------

    def _post(self, solver: str, data: bytes, params: dict) -> SQBMResult:
        if solver not in self._SOLVERS:
            raise ValueError(f"Solver '{solver}' is not valid. Choose from: {self._SOLVERS}")

        url = f"{self.base_url}/solver/{solver}"
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}

        resp = self._session.post(url, data=data, params=params, timeout=self.timeout)
        # Surface the server's explanation on error: SQBM+ returns the reason
        # ("unknown format", size/variable limits exceeded, ...) in the body,
        # which raise_for_status() would otherwise discard.
        if not resp.ok:
            detail = resp.text.strip()
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {url}"
                + (f"\nServer said: {detail}" if detail else ""),
                response=resp,
            )

        body = resp.json()
        return SQBMResult(
            id=body.get("id", ""),
            time=body.get("time", 0.0),
            wait=body.get("wait", 0.0),
            message=body.get("message", ""),
            runs=body.get("runs", 0),
            value=body.get("value", float("inf")),
            result=body.get("result", []),
            param=body.get("param", {}),
            count=body.get("count", 0),
            others=body.get("others", []),
        )

    # ------------------------------------------------------------------
    # Solver QUBO  (numpy matrix)
    # ------------------------------------------------------------------

    def solve_qubo(
        self,
        Q: np.ndarray,
        *,
        problem_format: Optional[str] = None,
        steps: Optional[int] = None,
        loops: Optional[int] = None,
        timeout: Optional[int] = None,
        maxwait: Optional[int] = None,
        target: Optional[float] = None,
        maxout: Optional[int] = None,
        algo: Optional[int] = None,
        dt: Optional[float] = None,
        C: Optional[float] = None,
    ) -> SQBMResult:
        """
        Solves a QUBO problem given a square numpy matrix.

        Args:
            Q:              QUBO matrix (numpy 2D square array).
            problem_format: Override the client's default encoding for this call
                            ("matrixmarket" or "hdf5"). None = use the instance default.
            steps:          Number of SB steps (0 = auto).
            loops:          Number of loops (0 = max).
            timeout:        Maximum solver computation time in seconds.
            maxwait:        Maximum queue wait time in seconds.
            target:         Stop when the objective value reaches this value.
            maxout:         Maximum number of returned solutions.
            algo:           Algorithm (15=bSB, 20=dSB, 25/30=escape local min, 0=auto).
            dt:             SB time step (0 = auto).
            C:              Parameter C (0 = auto).

        Returns:
            SQBMResult with the found solution.
        """
        data = self._encode_matrix(Q, problem_format)
        params = dict(
            steps=steps, loops=loops, timeout=timeout,
            maxwait=maxwait, target=target, maxout=maxout,
            algo=algo, dt=dt, C=C,
        )
        return self._post("qubo", data, params)

    # ------------------------------------------------------------------
    # Solver QUBO  (pre-built file)
    # ------------------------------------------------------------------

    def solve_qubo_file(
        self,
        filepath: str,
        *,
        steps: Optional[int] = None,
        loops: Optional[int] = None,
        timeout: Optional[int] = None,
        maxwait: Optional[int] = None,
        target: Optional[float] = None,
        maxout: Optional[int] = None,
        algo: Optional[int] = None,
        dt: Optional[float] = None,
        C: Optional[float] = None,
    ) -> SQBMResult:
        """
        Solves a QUBO problem from a file (MatrixMarket or HDF5). The server
        auto-detects the format from the file content, so no format flag is needed.
        """
        with open(filepath, "rb") as f:
            data = f.read()
        params = dict(
            steps=steps, loops=loops, timeout=timeout,
            maxwait=maxwait, target=target, maxout=maxout,
            algo=algo, dt=dt, C=C,
        )
        return self._post("qubo", data, params)

    # ------------------------------------------------------------------
    # Generic solver (qplib, pubo, tsp, shift, qap)
    # ------------------------------------------------------------------

    def solve_file(
        self,
        solver: str,
        filepath: str,
        *,
        content_type: str = "application/octet-stream",
        **params,
    ) -> SQBMResult:
        """
        Solves a generic problem from a file for any solver.

        Args:
            solver:       Solver name ('qplib', 'pubo', 'tsp', 'shift', 'qap').
            filepath:     Input file path.
            content_type: Content-Type header (use 'application/json' for shift).
            **params:     Optional query parameters (steps, loops, timeout, ...).

        Returns:
            SQBMResult with the found solution.
        """
        with open(filepath, "rb") as f:
            data = f.read()

        self._session.headers.update({"Content-Type": content_type})
        try:
            return self._post(solver, data, params)
        finally:
            self._session.headers.update({"Content-Type": "application/octet-stream"})

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Checks if the SQBM+ server is active.

        Returns:
            True if the server responds with status 'pass', False otherwise.
        """
        url = f"{self.base_url}/healthcheck"
        try:
            resp = self._session.get(url, timeout=10)
            return resp.status_code == 200 and resp.json().get("status") == "pass"
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Version check
    # ------------------------------------------------------------------

    def version(self) -> str:
        """
        Returns the version of SQBM+ in use.

        Returns:
            Version string, e.g. '2.1.0'.
        """
        url = f"{self.base_url}/version"
        resp = self._session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("version", "unknown")

    def __repr__(self) -> str:
        return f"SQBMClient(base_url='{self.base_url}', problem_format='{self.problem_format}')"


# ---------------------------------------------------------------------------
# High-level WFLO solver  (same interface as solve_wflo_ortools / solve_wflo_dwave)
# ---------------------------------------------------------------------------

def solve_wflo_sqbm(
    wake_loss_matrix,
    invalid_pairs,
    Q,
    n_turbines: int = 81,
    host: str = "localhost",
    port: int = 8000,
    problem_format: str = "hdf5",
    http_timeout: int = 120,
    steps: Optional[int] = None,
    loops: Optional[int] = None,
    timeout: Optional[int] = None,
    maxwait: Optional[int] = None,
    target: Optional[float] = None,
    maxout: Optional[int] = None,
    algo: Optional[int] = None,
    dt: Optional[float] = None,
    C: Optional[float] = None,
) -> tuple[np.ndarray, SQBMResult]:
    """
    Risolve il WFLO via QUBO usando SQBM+ (REST API).

    Parameters
    ----------
    wake_loss_matrix : ndarray (n, n)
        Matrice simmetrica dei wake loss fisici (per greedy repair e calcolo finale).
    invalid_pairs : ndarray (M, 2)
        Coppie che violano la distanza minima (per greedy repair).
    Q : ndarray (n, n)
        Matrice QUBO pre-costruita.
    n_turbines : int
        Numero esatto di turbine da posizionare.
    host : str
        IP o hostname del server SQBM+.
    port : int
        Porta del server SQBM+ (default 8000).
    problem_format : str
        Encoding per l'invio della matrice: 'hdf5' (default) o 'matrixmarket'.
    http_timeout : int
        Timeout HTTP in secondi (trasporto, non budget di calcolo del solver).
    steps : int, optional
        Numero di step SB (0 = auto).
    loops : int, optional
        Numero di loop (0 = massimo).
    timeout : int, optional
        Budget di calcolo del solver in secondi.
    maxwait : int, optional
        Attesa massima in coda in secondi.
    target : float, optional
        Ferma quando il valore obiettivo raggiunge questo valore.
    maxout : int, optional
        Numero massimo di soluzioni restituite.
    algo : int, optional
        Algoritmo (15=bSB, 20=dSB, 25/30=escape local min, 0=auto).
    dt : float, optional
        Time step SB (0 = auto).
    C : float, optional
        Parametro C (0 = auto).

    Returns
    -------
    solution : ndarray (n,) int
        Vettore binario delle turbine selezionate.
    sqbm_result : SQBMResult
        Risultato grezzo restituito dal server SQBM+.
    """
    from solvers.dwave import _greedy_repair

    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]

    # Ensure the HTTP transport timeout is always larger than the solver budget.
    # Without this, requests closes the socket exactly when the server is about
    # to reply, producing a ReadTimeout.
    if timeout is not None:
        http_timeout = max(http_timeout, timeout + 60)

    print(f"n = {n} variabili, SQBM+ @ {host}:{port} ({problem_format})")

    client = SQBMClient(
        host=host,
        port=port,
        timeout=http_timeout,
        problem_format=problem_format,
    )

    if not client.health_check():
        raise RuntimeError(
            f"Server SQBM+ non raggiungibile a {host}:{port}. "
            "Avvia il server oppure controlla host/port."
        )

    t0 = time.time()
    sqbm_result = client.solve_qubo(
        Q,
        steps=steps,
        loops=loops,
        timeout=timeout,
        maxwait=maxwait,
        target=target,
        maxout=maxout,
        algo=algo,
        dt=dt,
        C=C,
    )
    elapsed = time.time() - t0
    print(f"Campionamento completato in {elapsed:.1f}s  (server: {sqbm_result.time:.1f}s)")

    solution = np.array(sqbm_result.result, dtype=int)

    if solution.sum() != n_turbines:
        print(
            f"Cardinalità {solution.sum()} ≠ {n_turbines}, avvio greedy repair..."
        )
        solution = _greedy_repair(solution, Q, invalid_pairs, n_turbines)

    L = np.asarray(wake_loss_matrix, dtype=float)
    sel = np.where(solution == 1)[0]
    wake_loss = float(L[np.ix_(sel, sel)].sum() / 2)

    print(f"Turbine selezionate: {solution.sum()}")
    print(f"Pairwise wake loss: {wake_loss:.4f}")

    return solution, sqbm_result


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Q = np.array([
        [ 1.,        -2.,        -0.77798043, -2.,        -0.03766832, -0.08344094,  0.,         0.,         0.        ],
        [-2.,         1.,        -2.,          0.,        -2.,         -0.03766832,  0.,         0.,         0.        ],
        [ 0.,        -2.,         1.,          0.,         0.,         -2.,          0.,         0.,         0.        ],
        [-2.,        -0.03766832, -0.08344094,  1.,        -2.,        -0.77798043, -2.,        -0.03766832, -0.08344094],
        [ 0.,        -2.,        -0.03766832, -2.,         1.,         -2.,          0.,        -2.,         -0.03766832],
        [ 0.,         0.,        -2.,          0.,        -2.,          1.,          0.,         0.,         -2.        ],
        [ 0.,         0.,         0.,         -2.,        -0.03766832, -0.08344094,  1.,        -2.,         -0.77798043],
        [ 0.,         0.,         0.,          0.,        -2.,         -0.03766832, -2.,         1.,         -2.        ],
        [ 0.,         0.,         0.,          0.,         0.,         -2.,          0.,        -2.,          1.        ],
    ])

    # Default is MatrixMarket. Switch the whole client to HDF5:
    #   client = SQBMClient(host="localhost", port=8000, problem_format="hdf5", timeout=600)
    client = SQBMClient(host="localhost", port=8000, timeout=600)

    print("Health check:", client.health_check())
    print("Version:     ", client.version())

    # Per-call override + a realistic solver time budget.
    result = client.solve_qubo(Q, problem_format="hdf5", timeout=10, algo=20)
    print(result)

    # --- Correctness check: both formats must agree on the same problem ---
    r_mm = client.solve_qubo(Q, problem_format="matrixmarket", timeout=10, algo=20)
    r_h5 = client.solve_qubo(Q, problem_format="hdf5", timeout=10, algo=20)
    print("value match:", np.isclose(r_mm.value, r_h5.value))