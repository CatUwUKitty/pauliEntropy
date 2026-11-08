#!/usr/bin/env python3
"""Submit and resume Pauli entropy batches on a PBS cluster."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path


BASELINE_RANGES = (
    (1, 570), (571, 680), (681, 750), (751, 805), (806, 850),
    (851, 890), (891, 920), (921, 950), (951, 975), (976, 1000),
)
# Each pair is the inclusive end of a range and its initial RAM in GB.
# Range widths are balanced using the measured p=0.05 solve times.
P_SWEEP_ENDS = (
    ("0.01", "04:00:00", ((1000, 32),)),
    ("0.10", "08:00:00", (
        (557, 32), (663, 32), (735, 32), (794, 64), (840, 64), (874, 64),
        (902, 64), (932, 64), (956, 96), (979, 96), (1000, 96),
    )),
    ("0.20", "08:00:00", (
        (455, 32), (571, 32), (644, 32), (694, 32), (743, 64), (787, 64),
        (823, 64), (855, 64), (879, 96), (900, 96), (924, 96), (947, 96),
        (964, 96), (983, 96), (1000, 96),
    )),
)
MEMORY_TIERS = {32: (32, 64), 64: (64, 96), 96: (96,)}
Job = namedtuple("Job", "key name start end p epsilon walltime memory_gb")
POLL_SECONDS = 60
JOB_ID = re.compile(r"[0-9]+(?:\.[\w.-]+)?(?:@[\w.-]+)?")
EXIT_UNREPORTED = "unreported"


def p_job(p, start, end, walltime, ram, epsilon=None):
    label = f"{100 * float(p):02.0f}"
    return Job(f"p{p}/{start}-{end}", f"P{label}_{start}_{end}", start, end,
               p, epsilon, walltime, MEMORY_TIERS[ram])


def batch_jobs(batch):
    if batch == "baseline":
        return [Job(f"{start}-{end}", f"Pauli_{start}_{end}", start, end,
                    None, None, "08:00:00", MEMORY_TIERS[32])
                for start, end in BASELINE_RANGES]
    if batch == "p-pilot":
        return [p_job(p, 1000, 1000, "01:00:00", 96, epsilon="0.1")
                for p in ("0.10", "0.20")]

    jobs = []
    for p, walltime, ranges in P_SWEEP_ENDS:
        start = 1
        for end, ram in ranges:
            jobs.append(p_job(p, start, end, walltime, ram))
            start = end + 1
    return jobs


def pbs(*args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=60)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{' '.join(args)} failed with status {result.returncode}. {detail}")
    return result.stdout


def require_history():
    settings = pbs("qstat", "-Bf")
    values = re.findall(r"^\s*job_history_enable\s*=\s*(\w+)", settings, re.MULTILINE)
    if not values or any(value.lower() != "true" for value in values):
        raise RuntimeError("PBS job history must be enabled and visible in qstat -Bf to start retries.")


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finished_status(job_id):
    output = json.loads(pbs("qstat", "-x", "-f", "-F", "json", job_id))
    job = output.get("Jobs", {}).get(job_id)
    if job is None:
        raise RuntimeError(f"PBS returned no history for {job_id}. Check that job before resuming.")
    if job.get("job_state") != "F":
        return None
    status = job.get("Exit_status")
    return EXIT_UNREPORTED if status in (None, "") else int(status)


def refresh_attempt(key, attempts, path, state):
    if not attempts or attempts[-1]["exit_status"] is not None:
        return
    attempt = attempts[-1]
    status = finished_status(attempt["job_id"])
    if status is not None:
        attempt["exit_status"] = status
        save_state(path, state)
        print(f"{key}: {attempt['job_id']} finished (exit status {status})", flush=True)


def submit_attempt(root, output_root, job, ram, attempts, path, state):
    output_dir = output_root / "shards" / job.key
    log_path = path.parent / f"{job.name}_{ram}gb.log"
    variables = [f"PAULI_MIN_N={job.start}", f"PAULI_MAX_N={job.end}",
                 f"PAULI_OUTPUT_DIR={output_dir.relative_to(root).as_posix()}"]
    if job.p is not None:
        variables.append(f"PAULI_P={job.p}")
    if job.epsilon is not None:
        variables.append(f"PAULI_COLLISION_EPSILON={job.epsilon}")
    variables.extend(variable for variable in ("MOSEKLM_LICENSE_FILE", "PAULI_VENV", "PAULI_MOSEK_THREADS")
                     if os.environ.get(variable))
    args = (
        "qsub", "-N", job.name, "-o", str(log_path),
        "-l", f"select=1:ncpus=12:mem={ram}gb", "-l", f"walltime={job.walltime}",
        "-v", ",".join(variables),
        "submit.pbs",
    )
    # Record intent before qsub. An interrupted submission must be reconciled
    # with PBS before attempting the same range again.
    attempt = {"ram_gb": ram, "job_id": None, "exit_status": None, "log": str(log_path)}
    attempts.append(attempt)
    save_state(path, state)
    job_id = pbs(*args).strip()
    if not JOB_ID.fullmatch(job_id):
        raise RuntimeError(f"Unexpected job ID for {job.name}. Check PBS before resuming.")
    attempt["job_id"] = job_id
    save_state(path, state)
    print(f"{job.name}: submitted {job_id} at {ram} GB", flush=True)


def monitor(root, output_root, jobs, path, state):
    while True:
        active = False
        for job in jobs:
            attempts = state[job.key]
            refresh_attempt(job.key, attempts, path, state)
            if attempts and attempts[-1]["exit_status"] is None:
                active = True
                continue
            if attempts and attempts[-1]["exit_status"] in (0, EXIT_UNREPORTED):
                continue
            previous_ram = attempts[-1]["ram_gb"] if attempts else 0
            ram = next((candidate for candidate in job.memory_gb if candidate > previous_ram), None)
            if ram is None:
                continue
            submit_attempt(root, output_root, job, ram, attempts, path, state)
            active = True

        if not active:
            break
        time.sleep(POLL_SECONDS)

    failed = [key for key, attempts in state.items() if attempts[-1]["exit_status"] != 0]
    print(f"Completed {len(jobs) - len(failed)} of {len(jobs)} ranges.", flush=True)
    if failed:
        print(f"Ranges needing a restart: {', '.join(failed)}. See {path} and the attempt logs.", flush=True)
    return int(bool(failed))


def restart_finished(jobs, path, state, restart_failed):
    restart = []
    for job in jobs:
        attempts = state[job.key]
        refresh_attempt(job.key, attempts, path, state)
        if not attempts:
            continue
        last = attempts[-1]
        if last["exit_status"] not in (None, 0) and (
                restart_failed or last["exit_status"] == EXIT_UNREPORTED
                or last["ram_gb"] >= job.memory_gb[-1]):
            restart.append(job.key)
    if not restart:
        return
    archive = Path(tempfile.mkdtemp(prefix="restart-", dir=str(path.parent)))
    shutil.copy2(path, archive / path.name)
    for key in restart:
        for attempt in state[key]:
            log = Path(attempt["log"])
            if log.is_file():
                shutil.copy2(log, archive / log.name)
        state[key] = []
    save_state(path, state)
    print(f"Restarting {len(restart)} ranges. Previous attempts are in {archive}.", flush=True)


def main():
    import fcntl

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("baseline", "p-pilot", "p-sweep"), default="baseline",
                        help="Use the p=0.05 baseline, two high-n pilots, or the 27-job multi-p sweep")
    restart_options = parser.add_mutually_exclusive_group()
    restart_options.add_argument("--retry-exhausted", action="store_true",
                                 help="Restart ranges that reached their RAM limit or have no exit status")
    restart_options.add_argument("--restart-failed", action="store_true",
                                 help="Restart all finished ranges without a successful exit")
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "submit.pbs").is_file():
        raise RuntimeError("Run from the cluster's pauliEntropy checkout containing submit.pbs.")
    jobs = batch_jobs(args.batch)
    output_root = root / "results"
    if args.batch != "baseline":
        output_root /= args.batch
    directory = output_root / "logs" / "auto-retry"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    with (directory / "controller.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A retry controller is already running for this batch.") from error

        state = {job.key: [] for job in jobs}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.keys() != state.keys():
                raise RuntimeError(f"Unexpected copy-count ranges in {path}.")
            state = saved
        for key, attempts in state.items():
            if attempts and attempts[-1]["job_id"] is None:
                raise RuntimeError(f"Submission for {key} is unresolved in {path}. Check PBS for the job "
                                   "before recording its ID or removing the pending attempt.")
        require_history()
        if args.retry_exhausted or args.restart_failed:
            restart_finished(jobs, path, state, args.restart_failed)
        return monitor(root, output_root, jobs, path, state)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
    except KeyboardInterrupt:
        sys.exit("Controller stopped. Restart it from the same checkout to resume monitoring.")
