"""Run checkpointed sweeps using the solver cells in the companion notebook."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "pauli_combined_entropy.ipynb"
RUNNER_CELLS = (
    "pauli-001",  # Imports
    "pauli-002",  # Default settings
    "pauli-003",  # Checkpoint and validation helpers
    "pauli-005",  # State and reference functions
    "pauli-007",  # Collision solver
    "pauli-010",  # Min-entropy solvers
    "pauli-cluster-cases",  # Checkpoint row builders
)


def positive_int(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("copy counts must be positive")
    return count


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    counts = parser.add_mutually_exclusive_group(required=True)
    counts.add_argument("--n", nargs="+", type=positive_int, help="Selected copy counts")
    counts.add_argument("--max-n", type=positive_int, help="Inclusive end of a dense sweep")
    parser.add_argument("--min-n", type=positive_int, help="Inclusive start with --max-n (default 1)")
    parser.add_argument("--calculations", nargs="+", choices=("collision", "min_entropy"),
                        default=("collision", "min_entropy"))
    parser.add_argument("--p", nargs="+", type=float)
    parser.add_argument("--collision-epsilon", nargs="+", type=float,
                        help="Collision epsilon, also used as min-entropy radius squared by default")
    parser.add_argument("--min-entropy-epsilon", nargs="+", type=float,
                        help="Override the min-entropy squared radius")
    parser.add_argument("--solver", type=str.upper, choices=("MOSEK", "CLARABEL", "SCS"))
    parser.add_argument("--tol", type=float)
    parser.add_argument("--max-collision-copies", type=positive_int)
    parser.add_argument("--max-scalar-min-copies", type=positive_int)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("PAULI_OUTPUT_DIR", ROOT / "results")))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    errors = parser.add_mutually_exclusive_group()
    errors.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    errors.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    parser.set_defaults(continue_on_error=True)
    args = parser.parse_args()
    if args.min_n is not None and (args.max_n is None or args.min_n > args.max_n):
        parser.error("--min-n requires --max-n and must be at most --max-n")
    return args


def load_solver_cells(output_dir, seeds):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code"}
    namespace = {"OUTPUT_DIR": output_dir, "SAVED_RESULTS_DIR": seeds}
    os.environ.setdefault("MPLBACKEND", "Agg")
    for cell_id in RUNNER_CELLS:
        if cell_id not in sources:
            raise ValueError(f"Missing solver cell {cell_id} in {NOTEBOOK.name}")
        exec(compile(sources[cell_id], f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)
    return namespace


def main():
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    seeds = ROOT / "pauli_entropy_results"
    if output_dir == seeds.resolve():
        raise SystemExit("Choose a writable output directory separate from the bundled checkpoints.")
    if not args.resume and any((output_dir / f"{calculation}_results.csv").exists()
                               for calculation in args.calculations):
        raise SystemExit("Use a fresh output directory with --no-resume.")

    ns = load_solver_cells(output_dir, seeds)
    ns.update(RESUME=args.resume, CONTINUE_ON_ERROR=args.continue_on_error)
    overrides = {
        "P_VALUES": args.p,
        "COLLISION_EPSILON_VALUES": args.collision_epsilon,
        "MIN_ENTROPY_EPSILON_VALUES": args.min_entropy_epsilon or args.collision_epsilon,
        "COLLISION_SOLVER": args.solver,
        "COLLISION_TOL": args.tol,
        "MAX_COLLISION_COPIES": args.max_collision_copies,
        "MAX_SCALAR_MIN_COPIES": args.max_scalar_min_copies,
    }
    ns.update({key: value for key, value in overrides.items() if value is not None})
    p_values = ns["checked_values"](ns["P_VALUES"], "P_VALUES")
    n_values = args.n if args.n else range(args.min_n or 1, args.max_n + 1)

    jobs = {
        "collision": (ns["COLLISION_EPSILON_VALUES"], ns["collision_case"], ns["MAX_COLLISION_COPIES"]),
        "min_entropy": (ns["MIN_ENTROPY_EPSILON_VALUES"], ns["min_entropy_case"], ns["MAX_SCALAR_MIN_COPIES"]),
    }
    for calculation in args.calculations:
        epsilon_values, _, limit = jobs[calculation]
        if max(n_values) > limit:
            raise SystemExit(f"{calculation}: copy counts must be at most {limit}")
        ns["checked_values"](epsilon_values, f"{calculation} epsilon values", "epsilon")
    if "collision" in args.calculations:
        ns["require_solver"](ns["COLLISION_SOLVER"])
        ns["solver_options"](ns["COLLISION_SOLVER"], ns["COLLISION_TOL"])
        if ns["COLLISION_SOLVER"] == "MOSEK":
            cp = ns["cp"]
            probe = cp.Variable(nonneg=True)
            cp.Problem(cp.Maximize(probe), [probe <= 1]).solve(
                solver="MOSEK", canon_backend=cp.SCIPY_CANON_BACKEND)

    reference = ns["pd"].DataFrame([
        dict(p=p, H_vN_bits=ns["pauli_conditional_entropy"](p),
             V_bits_squared=ns["pauli_information_variance"](p),
             Hmin_unsmoothed_bits=ns["unsmoothed_min_entropy"](p))
        for p in p_values
    ])
    reference.to_csv(output_dir / "reference_results.csv", index=False)
    print(f"Python packages: CVXPY {ns['cp'].__version__}, NumPy {ns['np'].__version__}")
    print(f"Output: {output_dir}")

    errors = False
    for calculation in dict.fromkeys(args.calculations):
        epsilon_values, solve_case, _ = jobs[calculation]
        cases = ns["quantum_grid"](epsilon_values, n_values)
        frame = ns["run_sweep"](calculation, cases, solve_case)
        failed = frame["run_status"].ne("ok")
        if calculation == "collision":
            failed |= ~frame.get("solver_status", ns["pd"].Series("", index=frame.index)).isin(
                ("optimal", "analytic_endpoint"))
        errors |= bool(failed.any())
        if failed.any():
            print(frame.loc[failed, ["p", "epsilon", "n", "run_status", "error"]].to_string(index=False))
    return int(errors)


if __name__ == "__main__":
    raise SystemExit(main())
