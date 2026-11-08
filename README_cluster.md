# Pauli entropy on a PBS cluster

`pauli_combined_entropy.ipynb` contains the scaled collision solver and the
bracketed min-entropy solver. `run_cluster.py` executes those solver cells
from the same notebook. The two CSVs in `pauli_entropy_results/` seed a new
run. Working checkpoints and plots default to `results/`.

## Prepare the cluster environment

Clone or pull this repository on the cluster, then open a terminal at the
repository root.
The launcher inherits the project, queue, CPU and memory settings from the
previous PBS export. Check those directives against the cluster allocation.

Set up the Python environment on the cluster before submitting a job. The
commands below use the Python module and virtual environment from the
existing cluster setup. Create the virtual environment if needed.

```bash
source /app1/ebenv Python-3.10.4
python -m venv ~/myenv
source ~/myenv/bin/activate
python -m pip install -r requirements.txt
```

Put the cluster's MOSEK license at `~/mosek/mosek.lic`, or set
`MOSEKLM_LICENSE_FILE` to its location before starting Python. To override
it for a PBS job, use `qsub -v MOSEKLM_LICENSE_FILE=/path/to/mosek.lic submit.pbs`.
Keep the license outside the repository.

## Run a test and submit

With the environment active, run a small licensed test:

```bash
export MOSEKLM_LICENSE_FILE="$HOME/mosek/mosek.lic"
python run_cluster.py --n 1 2 --collision-epsilon 0 0.01 --output-dir results/smoke
```

The default PBS job evaluates all seven saved smoothing settings at
`n=250, 500, 750, 1000`:

```bash
qsub submit.pbs
```

For a dense sweep, request the inclusive upper copy count and a wall time
allowed by your queue:

```bash
qsub -l walltime=48:00:00 -v PAULI_MAX_N=1000 submit.pbs
```

The dense run starts at `n=1` by default, so it also refreshes older collision
values in the selected grid. Use `PAULI_MIN_N` to submit an inclusive slice
with `PAULI_MAX_N`. The wall time is an allocation example. Measure the
sample job on the cluster before sizing a dense run. The pilot at `n=1000`
is the largest tested range for this numerical implementation.

For a direct min-entropy sweep or other selections, use the runner:

```bash
python run_cluster.py --max-n 1000 --calculations min_entropy
python run_cluster.py --n 250 500 --collision-epsilon 1e-6 0.1
python run_cluster.py --help
```

## Run 1 through 1000 with automatic RAM retries

On the cluster login node, pull the latest version and run the retry controller
from the repository root. Use it in place of the manual submission loop.
Wait for any earlier Pauli jobs for these ranges to finish before starting.

```bash
git pull --ff-only origin main
qstat -Bf | grep job_history_enable
python3 -u submit_pauli_retries.py
```

PBS must expose `job_history_enable = True` so the controller can read each
finished job's exit status. The controller submits ten non-overlapping ranges,
each with 12 cores, eight hours of walltime and an initial 32 GB request.
A nonzero PBS exit status triggers a retry at 64 GB. The queue must allow
both memory requests. There is at most one active attempt per range.
Existing 20 GB attempts are monitored through completion. If one fails,
its next attempt requests 32 GB.

Run the controller in a persistent terminal on the login node if your site
allows it. Running the same command again resumes monitoring from
`results/logs/auto-retry/state.json`. Each attempt has a log named for its
range and RAM request in that directory. Stop the controller before
intentionally cancelling a job. An interrupted submission with an unknown job
ID requires checking PBS before recovering the pending entry.

PBS can report a finished job without an exit status, for example after
cancelling a queued job. The controller records it as unreported and waits
for an explicit restart of that range while monitoring the other jobs.

After cancelling the batch or fixing a failure, stop the old controller
and run `python3 -u submit_pauli_retries.py --restart-failed`. This checks
tracked job IDs, then restarts finished ranges with a nonzero or unreported
exit status at 32 GB. Use `--retry-exhausted` to limit the restart to 64 GB
failures and finished jobs with an unreported exit status. Both options
archive prior attempt logs and reuse CSV checkpoints. Successful and active
ranges keep their recorded state.

Each range writes to its own `results/shards/START-END/` directory. Retries
reuse that directory. To assemble the full sweep, take each shard's rows
within its assigned copy-count range before merging the CSVs for plotting.

## Sweep p = 0.01, 0.10, and 0.20

From the cluster checkout, pull the latest version and run the two
high-memory pilot jobs. Each runs both entropies at n = 1000 and
epsilon = 0.1. The queue must accept a 96 GB request for these jobs.

```bash
git pull --ff-only origin main
python3 -u submit_pauli_retries.py --batch p-pilot
```

After the pilots finish successfully, submit the full sweep. Run the
controller in a persistent session on the login node.

```bash
python3 -u submit_pauli_retries.py --batch p-sweep
```

The sweep covers every n from 1 through 1000 for both entropies and all
seven smoothing settings. It submits one job at p = 0.01, eleven at
p = 0.10, and fifteen at p = 0.20. The ranges were balanced using
timings from the completed p = 0.05 sweep. The p = 0.01 job requests
four hours and the other jobs request eight hours.

Each range starts at 32, 64, or 96 GB according to its largest retained
collision matrix. Failed 32 GB jobs retry at 64 GB, and failed jobs that
start at 64 GB retry at 96 GB. A failed attempt at the last tier needs
review. PBS controls how many of the 27 jobs run at a time.

The pilot and sweep keep independent state. Rerun the same command to
resume its batch. Sweep logs and state are in `results/p-sweep/logs/auto-retry/`,
and CSV checkpoints are in `results/p-sweep/shards/pVALUE/START-END/`.
For plotting, select the assigned p and copy-count range from each shard
before joining the collision and min-entropy CSVs.

## Resume and plot

Submit the same job again to resume. The runner reuses checkpoints with the
same parameters, solver and tolerance. It recomputes requested collision rows
from the older formulation and retries failed, inaccurate or flagged rows.
Each solve is written to the working CSV before moving to the next case.
Use one job per working output directory. For concurrent jobs, assign
different directories with `--output-dir` or `PAULI_OUTPUT_DIR`.

Open the notebook from the repository root and run all cells to plot saved
results. It prefers the working CSVs in `results/` when present, then falls
back to the bundled checkpoints. A difference point needs both entropies at
the same `p`, `epsilon` and `n`. For plotting results on another machine,
copy the working CSVs to that checkout's `results/` folder. To start a
cluster job from newer local checkpoints, copy `collision_results.csv` and
`min_entropy_results.csv` into the cluster checkout's `results/` before
submitting. Otherwise the runner starts from the checked-in seed CSVs.

After a job finishes, use `python push_results.py --results-dir results`
to publish its CSVs and generated plots from the cluster checkout.
