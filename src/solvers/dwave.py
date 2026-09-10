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

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    t = time.time() - t0
    print(f"BQM: {n} variabili, {bqm.num_interactions:,} interazioni ({t:.1f}s)")
    return bqm


def _get_sampler_and_kwargs(sampler_type, num_reads, num_sweeps, beta_range, time_limit_s, dwave_token, seed):
    if sampler_type == "simulated_annealing":
        if not _HAS_SAMPLERS:
            raise ImportError(
                "dwave-samplers non installato. Esegui: pip install dwave-samplers>=1.3"
            )
        sampler = SimulatedAnnealingSampler()
        kwargs = {
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "beta_range": beta_range,
        }
        if seed is not None:
            kwargs["seed"] = seed
        return sampler, kwargs

    elif sampler_type == "hybrid":
        if not _HAS_SYSTEM:
            raise ImportError(
                "dwave-system non installato. Esegui: pip install dwave-system"
            )
        sampler = LeapHybridSampler(token=dwave_token)
        kwargs = {"time_limit": time_limit_s}
        return sampler, kwargs

    elif sampler_type == "qpu":
        if not _HAS_SYSTEM:
            raise ImportError(
                "dwave-system non installato. Esegui: pip install dwave-system"
            )
        sampler = EmbeddingComposite(DWaveSampler(token=dwave_token))
        kwargs = {"num_reads": num_reads}
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
    print(f"Campioni con {n_turbines} turbine: {feasible_count}/{len(record)}")

    if feasible_count > 0:
        feasible_energies = np.where(feasible_mask, record.energy, np.inf)
        best_idx = int(np.argmin(feasible_energies))
        return record.sample[best_idx], False
    else:
        best_idx = int(np.argmin(record.energy))
        return record.sample[best_idx], True


def _greedy_repair(x, Q, invalid_pairs, n_turbines):
    x = x.copy().astype(int)

    invalid_adj = {}
    for i, j in invalid_pairs:
        invalid_adj.setdefault(int(i), set()).add(int(j))
        invalid_adj.setdefault(int(j), set()).add(int(i))

    while x.sum() > n_turbines:
        selected = np.where(x == 1)[0]
        Qx = Q @ x
        # delta_remove(i) = energy change when removing i:
        # Q[i,i]*1->0 and cross terms -2*(Q[i,:]@x) but exclude Q[i,i] already in Qx
        deltas = Q[selected, selected] - 2 * Qx[selected] + 2 * Q[selected, selected]
        # Simpler: delta = -Q[i,i] - 2*(Q[i,:]@x - Q[i,i]) = Q[i,i] - 2*(Q[i,:]@x)
        deltas = Q[selected, selected] - 2 * Qx[selected]
        worst = selected[int(np.argmin(deltas))]
        x[worst] = 0

    while x.sum() < n_turbines:
        unselected = np.where(x == 0)[0]
        selected_set = set(np.where(x == 1)[0].tolist())
        candidates = [
            i for i in unselected
            if not (invalid_adj.get(i, set()) & selected_set)
        ]
        if not candidates:
            break
        candidates = np.array(candidates)
        Qx = Q @ x
        deltas = Q[candidates, candidates] + 2 * Qx[candidates]
        best = candidates[int(np.argmin(deltas))]
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
):
    """
    Risolve il WFLO via QUBO usando un sampler D-Wave o SimulatedAnnealing.

    Parameters
    ----------
    wake_loss_matrix : ndarray (n, n)
        Matrice simmetrica dei wake loss fisici (usata per greedy repair e
        calcolo finale del wake loss).
    invalid_pairs : ndarray (M, 2)
        Coppie che violano la distanza minima (usate nel greedy repair).
    Q : ndarray (n, n)
        Matrice QUBO pre-costruita (da build_qubo_from_wake_matrix_optimized).
        Accettata direttamente per evitare di duplicare la logica di costruzione.
    n_turbines : int
        Numero esatto di turbine da posizionare.
    sampler_type : str
        'simulated_annealing' | 'hybrid' | 'qpu'
    num_reads : int
        Numero di campioni (per SA e QPU).
    num_sweeps : int
        Passi di sweep per campione (solo SA).
    beta_range : tuple (beta_min, beta_max)
        Range inverso-temperatura per SA. Default (1e-6, 1.0) calibrato per
        lambda_cardinality=1e6: beta_min=1e-6 → exp(-1)≈37% accettazione
        violazione cardinalità; beta_max=1.0 → sensibile a ~1 MWh di wake loss.
        Se usi lambda_cardinality diverso, ricalibra come (1/lambda, 1.0).
    time_limit_s : int
        Limite di tempo in secondi (solo hybrid).
    dwave_token : str or None
        Token API D-Wave Leap (solo hybrid/qpu).
    seed : int or None
        Seed per riproducibilità (solo SA).

    Returns
    -------
    solution : ndarray (n,) int
        Vettore binario delle turbine selezionate.
    sampleset : dimod.SampleSet
        Tutti i campioni restituiti dal sampler.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    print(f"n = {n} variabili, sampler_type = {sampler_type!r}")

    bqm = _qubo_to_bqm(Q)
    sampler, kwargs = _get_sampler_and_kwargs(
        sampler_type, num_reads, num_sweeps, beta_range, time_limit_s, dwave_token, seed
    )

    t0 = time.time()
    sampleset = sampler.sample(bqm, **kwargs)
    elapsed = time.time() - t0
    print(f"Campionamento completato in {elapsed:.1f}s")

    best_sample, needs_repair = _select_best_feasible(sampleset, n_turbines)

    solution = np.array([best_sample[i] for i in range(n)], dtype=int)

    if needs_repair:
        print(f"Nessun campione feasible trovato, avvio greedy repair...")
        solution = _greedy_repair(solution, Q, invalid_pairs, n_turbines)

    L = np.asarray(wake_loss_matrix, dtype=float)
    wake_loss = float(
        sum(
            L[i, j]
            for i in range(n)
            for j in range(i + 1, n)
            if solution[i] == 1 and solution[j] == 1
        )
    )

    sel = np.where(solution == 1)[0]
    wake_loss = float(L[np.ix_(sel, sel)].sum() / 2)

    print(f"Turbine selezionate: {solution.sum()}")
    print(f"Pairwise wake loss: {wake_loss:.4f}")

    return solution, sampleset
