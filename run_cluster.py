# Generated from pauli_combined_entropy.ipynb; edit the notebook and export again.
import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from functools import lru_cache
from math import comb, lgamma
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

if 'MOSEKLM_LICENSE_FILE' not in os.environ:
    for candidate in (Path.home() / 'mosek' / 'mosek.lic', Path.home() / '.mosek' / 'mosek.lic'):
        if candidate.is_file():
            os.environ['MOSEKLM_LICENSE_FILE'] = str(candidate)
            break

DEFAULT_CALCULATIONS = ['collision', 'min_entropy']

P_VALUES = [0.05]
COLLISION_EPSILON_VALUES = [0.1, 0.01, 0.001, 0.0001, 1e-05, 0.0]
MIN_ENTROPY_EPSILON_VALUES = list(COLLISION_EPSILON_VALUES)
n = 100
COLLISION_N_VALUES = list(range(1, n + 1))
MIN_ENTROPY_N_VALUES = list(range(1, n + 1))
COLLISION_SOLVER = 'MOSEK'
COLLISION_TOL = 1e-05
MAX_COLLISION_COPIES = 100
MAX_BASELINE_COPIES = 2
MAX_SCALAR_MIN_COPIES = 100
SOLVER_VERBOSE = False
RESUME = True
CONTINUE_ON_ERROR = False
RUN_VALIDATION = False
VALIDATION_N_VALUES = [1, 2]
VALIDATION_P_VALUES = [0.0, 0.05, 0.25, 0.5, 0.95, 1.0]
VALIDATION_EPSILON_VALUES = [0.0, 0.01, 0.1]
VALIDATION_RTOL = 0.0003
VALIDATION_ATOL = 2e-07
OUTPUT_DIR = Path('pauli_entropy_results')
MODEL_VERSION = 'pauli-pure-cq-v3-single-radius'

def checked_values(values, name, kind='probability'):
    values = list(values)
    if not values:
        raise ValueError(f'{name} must contain at least one value.')
    for value in values:
        if kind == 'copies':
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must contain positive integers.')
        elif not np.isfinite(value) or not 0 <= value <= 1 or (kind == 'epsilon' and value == 1):
            raise ValueError(f'Invalid value {value!r} in {name}.')
    return list(dict.fromkeys(values))

def require_solver(solver):
    if solver.upper() not in cp.installed_solvers():
        raise RuntimeError(f'Solver {solver} is unavailable: {cp.installed_solvers()}')

def solver_options(solver, tol):
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be positive and finite.')
    solver = solver.upper()
    if solver == 'CLARABEL':
        return dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=500)
    if solver == 'SCS':
        return dict(eps=tol, max_iters=200000)
    if solver == 'MOSEK':
        return dict(mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol})
    return {}

def checkpoint_key(calculation, case):
    payload = dict(model_version=MODEL_VERSION, calculation=calculation, **case)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

def write_checkpoint(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + '.tmp')
    pd.DataFrame(list(records.values())).to_csv(temporary_path, index=False, float_format='%.17g')
    os.replace(temporary_path, path)

def run_sweep(calculation, grid, solve_case):
    cases = list(grid)
    if not cases:
        raise ValueError(f'{calculation}: the parameter grid is empty.')
    path = Path(OUTPUT_DIR) / f'{calculation}_results.csv'
    records = {}
    if RESUME and path.exists():
        previous = pd.read_csv(path, float_precision='round_trip')
        if 'case_key' not in previous or 'run_status' not in previous:
            raise ValueError(f'Unrecognized checkpoint schema: {path}')
        if 'model_version' in previous:
            previous = previous.loc[previous['model_version'] == MODEL_VERSION]
        if 'error' in previous:
            previous['error'] = previous['error'].fillna('')
        records = {row['case_key']: row for row in previous.to_dict('records')}
    current_rows = []
    for index, case in enumerate(cases, 1):
        key = checkpoint_key(calculation, case)
        if key in records and records[key].get('run_status') == 'ok':
            row = records[key]
            row.update(case)
            print(f'[{index}/{len(cases)}] {calculation}: reused {case}')
        else:
            print(f'[{index}/{len(cases)}] {calculation}: solving {case}', flush=True)
            started = time.perf_counter()
            row = dict(case_key=key, model_version=MODEL_VERSION, **case)
            try:
                row.update(solve_case(case))
                row.update(run_status='ok', error='', wall_seconds=time.perf_counter() - started)
            except Exception as error:
                row.update(run_status='error', error=f'{type(error).__name__}: {error}', wall_seconds=time.perf_counter() - started)
                records[key] = row
                write_checkpoint(records, path)
                print(f"  Failed: {row['error']}", flush=True)
                if not CONTINUE_ON_ERROR:
                    raise
            records[key] = row
            write_checkpoint(records, path)
        current_rows.append(row)
    write_checkpoint(records, path)
    frame = pd.DataFrame(current_rows)
    print(f"{calculation}: {(frame['run_status'] == 'ok').sum()}/{len(frame)} successful; {path}")
    return frame

def successful_results(frame, name):
    if frame is None or frame.empty:
        print(f'{name}: no results available.')
        return pd.DataFrame()
    good = frame.loc[frame['run_status'] == 'ok'].copy()
    if good.empty:
        print(f'{name}: no successful results. Inspect the error column in its CSV.')
    return good

def print_result_table(frame, columns):
    selected = [column for column in columns if column in frame.columns]
    selected += [column for column in ('run_status', 'error') if column in frame.columns]
    print(frame[selected].to_string(index=False))

def validate_parameters(p, n, epsilon=None, matrix=False, collision=False):
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Require 0 <= p <= 1.')
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer.')
    if epsilon is not None and (not np.isfinite(epsilon) or not 0 <= epsilon < 1):
        raise ValueError('Require 0 <= smoothing parameter < 1.')
    limit = MAX_BASELINE_COPIES if matrix else MAX_COLLISION_COPIES if collision else MAX_SCALAR_MIN_COPIES
    if n > limit:
        raise ValueError(f'n={n} exceeds allocation/numerical guard {limit}.')

def quantum_grid(epsilon_values, n_values):
    for p, n, epsilon in itertools.product(checked_values(P_VALUES, 'P_VALUES'), checked_values(n_values, 'copy counts', 'copies'), checked_values(epsilon_values, 'epsilon values', 'epsilon')):
        yield dict(p=float(p), epsilon=float(epsilon), n=int(n), solver=COLLISION_SOLVER, tol=COLLISION_TOL)

def pauli_conditional_states(p):
    validate_parameters(p, 1)
    v = np.array([np.sqrt(p), np.sqrt(1 - p)])
    z = np.array([1.0, -1.0])
    return (np.outer(v, v), np.outer(v * z, v * z))

def generate_pauli_cq_blocks(p, n=1):
    validate_parameters(p, n, matrix=True)
    blocks = [r / 2 for r in pauli_conditional_states(p)]
    single = list(blocks)
    for _ in range(n - 1):
        blocks = [np.kron(b, s) for b in blocks for s in single]
    return blocks

def seed_vector(p, n):
    validate_parameters(p, n, matrix=True)
    v = np.ones(1)
    for _ in range(n):
        v = np.kron(v, [np.sqrt(p), np.sqrt(1 - p)])
    return v

def h2(p):
    validate_parameters(p, 1)
    return -sum((t * math.log2(t) for t in (p, 1 - p) if t > 0))

def pauli_conditional_entropy(p):
    return 1 - h2(p)

def pauli_information_variance(p):
    validate_parameters(p, 1)
    return 0.0 if p in (0, 1) else p * (1 - p) * math.log2((1 - p) / p) ** 2

def unsmoothed_min_entropy(p, n=1):
    validate_parameters(p, n)
    return -n * math.log2((1 + 2 * math.sqrt(p * (1 - p))) / 2)

@lru_cache(maxsize=128)
def hamming_vectors(p, n):
    """Immutable sector weights, shared across epsilon values and both entropies."""
    k = np.arange(n + 1)
    log_m = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    u = np.exp(0.5 * binom.logpmf(k, n, p))
    u /= np.linalg.norm(u)
    c = np.exp(0.5 * (log_m - n * math.log(2)))
    inv_sqrt_m = np.exp(-0.5 * log_m)
    for array in (u, c, inv_sqrt_m):
        array.setflags(write=False)
    return (u, c, inv_sqrt_m)

@lru_cache(maxsize=1)
def collision_model(p, n):
    """One reusable model; keeping only one bounds compiled-model memory."""
    size = n + 1
    scale = 2.0 ** (n * h2(p))
    u, _, inv_sqrt_m = hamming_vectors(p, n)
    S = cp.Variable((size, size), symmetric=True)
    T, K = (cp.Variable(nonneg=True), cp.Variable(nonneg=True))
    epsilon_parameter = cp.Parameter(nonneg=True)
    scaled_columns = cp.multiply(S, math.sqrt(scale) * inv_sqrt_m[None, :])
    cone_vectors = cp.vstack([2 * scaled_columns, (K - 1) * np.ones((1, size))])
    constraints = [S >> 0, T * np.eye(size) - S >> 0, cp.SOC((K + 1) * np.ones(size), cone_vectors, axis=0), cp.SOC(K + 1, cp.hstack([2 * T * math.sqrt(scale / 2 ** n), K - 1]))]
    problem = cp.Problem(cp.Maximize(2 * cp.sum(cp.multiply(np.outer(u, u), S)) - 2 * epsilon_parameter * T - K), constraints)
    assert problem.is_dcp(dpp=True)
    return (problem, epsilon_parameter, S, T, K, inv_sqrt_m, scale)

def solve_pauli_collision(p=0.05, epsilon=0.001, n=1, solver='CLARABEL', tol=1e-08, verbose=False):
    validate_parameters(p, n, epsilon, collision=True)
    require_solver(solver)
    started = time.perf_counter()
    p = float(min(p, 1 - p))
    if p in (0.0, 0.5):
        H = (n if p == 0.0 else 0.0) - 2 * math.log1p(-epsilon) / math.log(2)
        return dict(q=2.0 ** (-H), H_bits=H, status='analytic_endpoint', solve_time=time.perf_counter() - started, compilation_seconds=0.0, solver_seconds=0.0, num_iters=0, matrix_dimension=n + 1, psd_violation=0.0, norm_violation=0.0)
    problem, epsilon_parameter, S, T, K, inv_sqrt_m, scale = collision_model(p, int(n))
    epsilon_parameter.value = float(epsilon)
    problem.solve(solver=solver.upper(), verbose=verbose, warm_start=True, enforce_dpp=True, canon_backend=cp.SCIPY_CANON_BACKEND, **solver_options(solver, tol))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'Collision solver: {problem.status}')
    if problem.status == cp.OPTIMAL_INACCURATE:
        warnings.warn('Collision solver returned optimal_inaccurate; inspect residuals.')
    a = scale * (S.value + S.value.T) / 2
    tv, kv, F = (scale * float(T.value), scale * float(K.value), scale * float(problem.value))
    eigenvalues = np.linalg.eigvalsh(a)
    psd_violation = float(max(0.0, -eigenvalues[0], eigenvalues[-1] - tv))
    norm_violation = max(0.0, float(np.max(np.sum((a * inv_sqrt_m[None, :]) ** 2, axis=0) - kv)))
    if max(psd_violation, norm_violation) > 100 * tol:
        warnings.warn('Collision feasibility residual exceeds 100*tolerance.')
    if not np.isfinite(F) or F <= 0:
        raise RuntimeError('Cannot compute collision entropy from this optimum.')
    return dict(q=F / 2 ** n, H_bits=n - math.log2(F), status=problem.status, solve_time=time.perf_counter() - started, compilation_seconds=float(problem.compilation_time or 0.0), solver_seconds=problem.solver_stats.solve_time, num_iters=problem.solver_stats.num_iters, matrix_dimension=n + 1, psd_violation=psd_violation, norm_violation=norm_violation)

def sanity_measured_smooth_collision_entropy(rho_blocks, epsilon=0.001, verbose=False, solver='CLARABEL', tol=1e-08):
    """
    Compute H_2^{epsilon,M,up}(X|E) using Eq. (B8).
    rho_blocks[x] = p(x) rho_{E|x}.
    """
    require_solver(solver)
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError('Require 0 <= epsilon < 1.')
    nX = len(rho_blocks)
    dE = rho_blocks[0].shape[0]
    I = np.eye(dE)
    B = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    C = [cp.Variable((dE, dE), hermitian=True) for _ in range(nX)]
    t = cp.Variable()
    k = cp.Variable()
    constraints = []
    for x in range(nX):
        constraints += [B[x] >> 0, B[x] << t * I, cp.bmat([[C[x], B[x]], [B[x], I]]) >> 0]
    constraints += [sum(C) << k * I]
    trace_term = sum((cp.real(cp.trace(B[x] @ rho_blocks[x])) for x in range(nX)))
    objective = cp.Maximize(2 * trace_term - 2 * epsilon * t - k)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver.upper(), verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND, **solver_options(solver, tol))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'Solver status: {problem.status}')
    Q_star = float(problem.value)
    if not np.isfinite(Q_star) or Q_star <= 0:
        raise RuntimeError('Cannot compute entropy from a nonpositive/nonfinite optimum.')
    H2 = -np.log2(Q_star)
    return {'entropy': H2, 'Q_star': Q_star, 't': t.value, 'k': k.value, 'status': problem.status}

def min_vectors(p, n):
    u, c, _ = hamming_vectors(float(min(p, 1 - p)), int(n))
    return (u, c)

def solve_pauli_min_entropy(p, n, radius_squared):
    validate_parameters(p, n, radius_squared)
    started = time.perf_counter()
    result = dict(_solve_pauli_min_cached(float(min(p, 1 - p)), int(n), float(radius_squared)))
    result['solve_time'] = time.perf_counter() - started
    return result

@lru_cache(maxsize=512)
def _solve_pauli_min_cached(p, n, radius_squared):
    validate_parameters(p, n, radius_squared)
    started = time.perf_counter()
    r2 = float(radius_squared)
    if r2 == 0:
        H = unsmoothed_min_entropy(p, n)
        return dict(H_bits=H, value=2.0 ** (-H), status='analytic_unsmoothed', solve_time=time.perf_counter() - started, fidelity_violation=0.0)
    p = min(p, 1 - p)
    f = math.sqrt(1 - r2)
    if p in (0.0, 0.5):
        H = (n if p == 0 else 0) - math.log2(1 - r2)
        return dict(H_bits=H, value=2.0 ** (-H), status='analytic_endpoint', solve_time=time.perf_counter() - started, fidelity_violation=0.0)
    u, c = min_vectors(p, n)
    if f <= u[0]:
        b = np.zeros(n + 1)
        b[0] = f / u[0]
        residual = max(0.0, f - float(u @ b))
    else:
        target_distance_squared = 2 * r2 / (1 + f)
        lo, hi = (0.0, float(u[0] / c[0]))
        for _ in range(110):
            mid = lo + (hi - lo) / 2
            if mid == lo or mid == hi:
                break
            raw = np.maximum(u - mid * c, 0.0)
            b = raw / np.linalg.norm(raw)
            distance_squared = float(np.dot(u - b, u - b))
            if distance_squared > target_distance_squared:
                hi = mid
            else:
                lo = mid
        raw = np.maximum(u - lo * c, 0.0)
        b = raw / np.linalg.norm(raw)
        residual = max(0.0, float(np.dot(u - b, u - b)) - target_distance_squared)
    amplitude = float(c @ b)
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise RuntimeError('Scalar min-entropy lost objective precision; reduce n.')
    H = -2 * math.log2(amplitude)
    return dict(H_bits=H, value=amplitude ** 2, status='scalar_threshold', solve_time=time.perf_counter() - started, fidelity_violation=residual)

def solve_pauli_min_socp(p, n, radius_squared, solver='CLARABEL', tol=1e-09, verbose=False):
    validate_parameters(p, n, radius_squared)
    u, c = min_vectors(p, n)
    if radius_squared == 0:
        return float(c @ u) ** 2
    b = cp.Variable(n + 1, nonneg=True)
    problem = cp.Problem(cp.Minimize(c @ b), [cp.norm(b, 2) <= 1, u @ b >= math.sqrt(1 - radius_squared)])
    problem.solve(solver=solver.upper(), verbose=verbose, **solver_options(solver, tol))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(problem.status)
    return float(problem.value) ** 2

def solve_pauli_min_baseline(p, n, radius_squared, solver='CLARABEL', tol=1e-09, verbose=False):
    validate_parameters(p, n, radius_squared, matrix=True)
    blocks = generate_pauli_cq_blocks(p, n)
    d = 2 ** n
    S = cp.Variable((d, d), symmetric=True)
    if radius_squared == 0:
        constraints = [S >> 0] + [S - B >> 0 for B in blocks]
    else:
        taus = [cp.Variable((d, d), symmetric=True) for _ in blocks]
        Ls = [cp.Variable((d, d)) for _ in blocks]
        constraints = [S >> 0]
        for B, tau, L in zip(blocks, taus, Ls):
            constraints += [S - tau >> 0, cp.bmat([[B, L], [L.T, tau]]) >> 0]
        constraints += [sum((cp.trace(t) for t in taus)) <= 1, sum((cp.trace(L) for L in Ls)) >= math.sqrt(1 - radius_squared)]
    problem = cp.Problem(cp.Minimize(cp.trace(S)), constraints)
    problem.solve(solver=solver.upper(), verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND, **solver_options(solver, tol))
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(problem.status)
    return float(problem.value)

def collision_case(case):
    r = solve_pauli_collision(**case, verbose=SOLVER_VERBOSE)
    return dict(q=r['q'], H_bits=r['H_bits'], bits_per_copy=r['H_bits'] / case['n'], solver_status=r['status'], solve_seconds=r['solve_time'], compilation_seconds=r['compilation_seconds'], solver_seconds=r['solver_seconds'], num_iters=r['num_iters'], matrix_dimension=r['matrix_dimension'], psd_violation=r['psd_violation'], norm_violation=r['norm_violation'])

def min_entropy_case(case):
    p, n, epsilon = (case['p'], case['n'], case['epsilon'])
    result = solve_pauli_min_entropy(p, n, epsilon)
    return dict(H_bits=result['H_bits'], bits_per_copy=result['H_bits'] / n, radius=math.sqrt(epsilon), radius_squared=epsilon, solver_status=result['status'], solve_seconds=result['solve_time'])

def parse_arguments():
    parser = argparse.ArgumentParser(description='Checkpointed Pauli entropy parameter sweeps.')
    parser.add_argument('--calculations', nargs='+', choices=['collision', 'min_entropy'],
                        default=DEFAULT_CALCULATIONS)
    parser.add_argument('--p', nargs='+', type=float, help='Pauli state probabilities')
    parser.add_argument('--collision-epsilon', nargs='+', type=float,
                        help='Collision epsilon; also sets min-entropy radius squared unless overridden')
    parser.add_argument('--min-entropy-epsilon', nargs='+', type=float,
                        help='Optional independent min-entropy squared purified-distance radii')
    parser.add_argument('--n', nargs='+', type=int, help='Copy counts for both entropies')
    parser.add_argument('--solver', type=str.upper, choices=['MOSEK', 'CLARABEL', 'SCS'])
    parser.add_argument('--tol', type=float)
    parser.add_argument('--max-collision-copies', type=int)
    parser.add_argument('--max-scalar-min-copies', type=int)
    parser.add_argument('--output-dir', help='Default: pauli_entropy_results, relative to working directory')
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument('--resume', dest='resume', action='store_true')
    resume.add_argument('--no-resume', dest='resume', action='store_false')
    errors = parser.add_mutually_exclusive_group()
    errors.add_argument('--fail-fast', dest='continue_on_error', action='store_false')
    errors.add_argument('--continue-on-error', dest='continue_on_error', action='store_true')
    parser.set_defaults(resume=None, continue_on_error=None)
    return parser.parse_args()


def apply_cli_arguments(args):
    global P_VALUES, COLLISION_EPSILON_VALUES, MIN_ENTROPY_EPSILON_VALUES
    global COLLISION_N_VALUES, MIN_ENTROPY_N_VALUES, COLLISION_SOLVER, COLLISION_TOL
    global MAX_COLLISION_COPIES, MAX_SCALAR_MIN_COPIES, OUTPUT_DIR, RESUME, CONTINUE_ON_ERROR
    if args.p is not None:
        P_VALUES = args.p
    if args.collision_epsilon is not None:
        COLLISION_EPSILON_VALUES = args.collision_epsilon
        MIN_ENTROPY_EPSILON_VALUES = list(args.collision_epsilon)
    if args.min_entropy_epsilon is not None:
        MIN_ENTROPY_EPSILON_VALUES = args.min_entropy_epsilon
    if args.n is not None:
        COLLISION_N_VALUES = list(args.n)
        MIN_ENTROPY_N_VALUES = list(args.n)
    if args.solver is not None:
        COLLISION_SOLVER = args.solver
    if args.tol is not None:
        COLLISION_TOL = args.tol
    if args.max_collision_copies is not None:
        MAX_COLLISION_COPIES = args.max_collision_copies
    if args.max_scalar_min_copies is not None:
        MAX_SCALAR_MIN_COPIES = args.max_scalar_min_copies
    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir)
    if args.resume is not None:
        RESUME = args.resume
    if args.continue_on_error is not None:
        CONTINUE_ON_ERROR = args.continue_on_error
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_arguments()
    apply_cli_arguments(args)
    any_errors = False
    reference_results = pd.DataFrame([dict(p=p, H_vN_bits=pauli_conditional_entropy(p), V_bits_squared=pauli_information_variance(p), Hmin_unsmoothed_bits=unsmoothed_min_entropy(p)) for p in P_VALUES])
    reference_results.to_csv(OUTPUT_DIR / 'reference_results.csv', index=False)
    if 'collision' in args.calculations:
        collision_results = run_sweep('collision', quantum_grid(COLLISION_EPSILON_VALUES, COLLISION_N_VALUES), collision_case)
        print_result_table(collision_results, ['p', 'epsilon', 'n', 'H_bits', 'bits_per_copy', 'radius'])
        if (collision_results['run_status'] == 'error').any():
            any_errors = True
    if 'min_entropy' in args.calculations:
        min_entropy_results = run_sweep('min_entropy', quantum_grid(MIN_ENTROPY_EPSILON_VALUES, MIN_ENTROPY_N_VALUES), min_entropy_case)
        print_result_table(min_entropy_results, ['p', 'epsilon', 'n', 'H_bits', 'bits_per_copy', 'radius'])
        if (min_entropy_results['run_status'] == 'error').any():
            any_errors = True
    return 1 if any_errors else 0

if __name__ == "__main__":
    sys.exit(main())
