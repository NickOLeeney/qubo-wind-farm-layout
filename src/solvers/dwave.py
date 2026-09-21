import time
import numpy as np
import dimod

try:
    from dwave.samplers import SimulatedAnnealingSampler
    _HAS_SAMPLERS = True
except ImportError:
    _HAS_SAMPLERS = False

try:
    from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
    _HAS_SYSTEM = True
except ImportError:
    _HAS_SYSTEM = False


def _qubo_to_bqm(Q):
    n = Q.shape[0]
    t0 = time.time()

    linear = {i: Q[i, i] for i in range(n)}

    rows, cols = np.triu_indices(n, k=1)
    quadratic = {
        (int(i), int(j)): Q[i, j] + Q[j, i]
        for i, j in zip(rows, cols)
        if Q[i, j] + Q[j, i] != 0.0
    }

    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        0.0,
        dimod.BINARY,
    )

    t = time.time() - t0
    print(
        f"BQM: {n} variabili, "
        f"{bqm.num_interactions:,} interazioni ({t:.1f}s)"
    )

    return bqm


def _validate_warm_start(warm_start, n):
    """
    Valida e normalizza un warm start binario.

    Accetta:
    - ndarray/list shape (n,)
    - ndarray/list shape (k, n), per più initial states

    Returns
    -------
    ndarray
        Shape (n,) oppure (k, n), dtype int8.
    """
    if warm_start is None:
        return None

    ws = np.asarray(warm_start)

    if ws.ndim not in (1, 2):
        raise ValueError(
            "warm_start deve avere shape (n,) oppure (k, n)."
        )

    if ws.ndim == 1:
        if ws.shape[0] != n:
            raise ValueError(
                f"warm_start ha lunghezza {ws.shape[0]}, "
                f"ma il QUBO ha {n} variabili."
            )
    else:
        if ws.shape[1] != n:
            raise ValueError(
                f"warm_start ha shape {ws.shape}, "
                f"ma ogni stato deve avere {n} variabili."
            )

    if not np.all(np.isin(ws, [0, 1])):
        raise ValueError(
            "warm_start deve contenere esclusivamente valori 0/1."
        )

    return ws.astype(np.int8, copy=False)


def _get_sampler_and_kwargs(
    sampler_type,
    num_reads,
    num_sweeps,
    beta_range,
    time_limit_s,
    dwave_token,
    seed,
    warm_start=None,
    warm_start_mode="tile",
):
    if sampler_type == "simulated_annealing":
        if not _HAS_SAMPLERS:
            raise ImportError(
                "dwave-samplers non installato. "
                "Esegui: pip install dwave-samplers>=1.3"
            )

        sampler = SimulatedAnnealingSampler()

        kwargs = {
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "beta_range": beta_range,
        }

        if seed is not None:
            kwargs["seed"] = seed

        if warm_start is not None:
            if warm_start_mode not in {"tile", "random", "none"}:
                raise ValueError(
                    "warm_start_mode deve essere "
                    "'tile', 'random' oppure 'none'."
                )

            kwargs["initial_states"] = warm_start
            kwargs["initial_states_generator"] = warm_start_mode

        return sampler, kwargs

    elif sampler_type == "hybrid":
        if warm_start is not None:
            raise ValueError(
                "warm_start è implementato in questa funzione "
                "solo per sampler_type='simulated_annealing'."
            )

        if not _HAS_SYSTEM:
            raise ImportError(
                "dwave-system non installato. "
                "Esegui: pip install dwave-system"
            )

        sampler = LeapHybridSampler(token=dwave_token)
        kwargs = {"time_limit": time_limit_s}

        return sampler, kwargs

    elif sampler_type == "qpu":
        if warm_start is not None:
            raise ValueError(
                "warm_start non è supportato dal normale forward annealing "
                "QPU in questa funzione. Per la QPU servirebbe "
                "reverse annealing."
            )

        if not _HAS_SYSTEM:
            raise ImportError(
                "dwave-system non installato. "
                "Esegui: pip install dwave-system"
            )

        sampler = EmbeddingComposite(
            DWaveSampler(token=dwave_token)
        )

        kwargs = {
            "num_reads": num_reads,
        }

        return sampler, kwargs

    else:
        raise ValueError(
            f"sampler_type sconosciuto: {sampler_type!r}. "
            "Usa 'simulated_annealing', 'hybrid', o 'qpu'."
        )


def _select_best_feasible(sampleset, n_turbines):
    record = sampleset.record

    counts = record.sample.sum(axis=1)
    feasible_mask = counts == n_turbines
    feasible_count = feasible_mask.sum()

    print(
        f"Campioni con {n_turbines} turbine: "
        f"{feasible_count}/{len(record)}"
    )

    if feasible_count > 0:
        feasible_energies = np.where(
            feasible_mask,
            record.energy,
            np.inf,
        )

        best_idx = int(
            np.argmin(feasible_energies)
        )

        return record.sample[best_idx], False

    best_idx = int(
        np.argmin(record.energy)
    )

    return record.sample[best_idx], True


def _greedy_repair(
    x,
    Q,
    invalid_pairs,
    n_turbines,
):
    x = x.copy().astype(int)

    invalid_adj = {}

    for i, j in invalid_pairs:
        invalid_adj.setdefault(
            int(i),
            set(),
        ).add(int(j))

        invalid_adj.setdefault(
            int(j),
            set(),
        ).add(int(i))

    while x.sum() > n_turbines:
        selected = np.where(x == 1)[0]

        Qx = Q @ x

        deltas = (
            Q[selected, selected]
            - 2 * Qx[selected]
        )

        worst = selected[
            int(np.argmin(deltas))
        ]

        x[worst] = 0

    while x.sum() < n_turbines:
        unselected = np.where(x == 0)[0]

        selected_set = set(
            np.where(x == 1)[0].tolist()
        )

        candidates = [
            i
            for i in unselected
            if not (
                invalid_adj.get(i, set())
                & selected_set
            )
        ]

        if not candidates:
            break

        candidates = np.asarray(
            candidates,
            dtype=int,
        )

        Qx = Q @ x

        deltas = (
            Q[candidates, candidates]
            + 2 * Qx[candidates]
        )

        best = candidates[
            int(np.argmin(deltas))
        ]

        x[best] = 1

    return x


def solve_wflo_dwave(
    wake_loss_matrix,
    invalid_pairs,
    Q,
    n_turbines=81,
    sampler_type="simulated_annealing",
    num_reads=1000,
    num_sweeps=10000,
    beta_range=(1e-6, 1.0),
    time_limit_s=20,
    dwave_token=None,
    seed=None,
    warm_start=None,
    warm_start_mode="tile",
):
    """
    Risolve il WFLO via QUBO usando D-Wave o Simulated Annealing.

    Parameters
    ----------
    wake_loss_matrix : ndarray (n, n)
        Matrice simmetrica dei wake loss fisici.

    invalid_pairs : ndarray (M, 2)
        Coppie che violano la distanza minima.

    Q : ndarray (n, n)
        Matrice QUBO pre-costruita.

    n_turbines : int
        Numero esatto di turbine.

    sampler_type : str
        'simulated_annealing' | 'hybrid' | 'qpu'

    num_reads : int
        Numero di annealing indipendenti.

    num_sweeps : int
        Numero di sweep per read, solo SA.

    beta_range : tuple
        Intervallo beta per SA.

    time_limit_s : int
        Time limit per LeapHybridSampler.

    dwave_token : str or None
        Token Leap.

    seed : int or None
        Seed SA.

    warm_start : array-like or None
        Stato iniziale binario.

        Può essere:
            shape (n,)
                singolo layout iniziale;

            shape (k, n)
                k layout iniziali.

        Supportato in questa implementazione per
        sampler_type='simulated_annealing'.

    warm_start_mode : {'tile', 'random', 'none'}
        Comportamento se il numero di warm-start states
        è inferiore a num_reads.

        'tile':
            ripete ciclicamente i warm starts forniti.

        'random':
            usa i warm starts forniti e genera casualmente
            gli initial states mancanti.

        'none':
            richiede che il numero di initial states sia
            sufficiente per num_reads.

    Returns
    -------
    solution : ndarray (n,) int
        Vettore binario finale.

    sampleset : dimod.SampleSet
        Campioni restituiti dal sampler.
    """
    Q = np.asarray(
        Q,
        dtype=float,
    )

    n = Q.shape[0]

    print(
        f"n = {n} variabili, "
        f"sampler_type = {sampler_type!r}"
    )

    warm_start = _validate_warm_start(
        warm_start,
        n,
    )

    if warm_start is not None:
        if warm_start.ndim == 1:
            print(
                "Warm start: "
                f"{warm_start.sum()} turbine"
            )
        else:
            cardinalities = warm_start.sum(axis=1)

            print(
                f"Warm starts: {len(warm_start)} stati | "
                f"cardinalità min={cardinalities.min()}, "
                f"max={cardinalities.max()}"
            )

    bqm = _qubo_to_bqm(Q)

    sampler, kwargs = _get_sampler_and_kwargs(
        sampler_type=sampler_type,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        time_limit_s=time_limit_s,
        dwave_token=dwave_token,
        seed=seed,
        warm_start=warm_start,
        warm_start_mode=warm_start_mode,
    )

    t0 = time.time()

    sampleset = sampler.sample(
        bqm,
        **kwargs,
    )

    elapsed = time.time() - t0

    print(
        f"Campionamento completato in {elapsed:.1f}s"
    )

    best_sample, needs_repair = _select_best_feasible(
        sampleset,
        n_turbines,
    )

    solution = np.array(
        [best_sample[i] for i in range(n)],
        dtype=int,
    )

    if needs_repair:
        print(
            "Nessun campione con cardinalità corretta trovato, "
            "avvio greedy repair..."
        )

        solution = _greedy_repair(
            solution,
            Q,
            invalid_pairs,
            n_turbines,
        )

    L = np.asarray(
        wake_loss_matrix,
        dtype=float,
    )

    sel = np.where(
        solution == 1
    )[0]

    wake_loss = float(
        L[np.ix_(sel, sel)].sum() / 2
    )

    spacing_violations = sum(
        solution[int(i)] == 1
        and solution[int(j)] == 1
        for i, j in invalid_pairs
    )

    print(
        f"Turbine selezionate: {solution.sum()}"
    )

    print(
        f"Spacing violations: {spacing_violations}"
    )

    print(
        f"Pairwise wake loss: {wake_loss:.4f}"
    )

    return solution, sampleset