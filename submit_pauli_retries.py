#!/usr/bin/env python3
"""Submit the ten Pauli ranges and retry failed PBS jobs at 20, 32, and 64 GB."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


RANGES = (
    (1, 570), (571, 680), (681, 750), (751, 805), (806, 850),
    (851, 890), (891, 920), (921, 950), (951, 975), (976, 1000),
)
MEMORY_GB = (20, 32, 64)
POLL_SECONDS = 60
JOB_ID = re.compile(r"[0-9]+(?:\.[\w.-]+)?(?:@[\w.-]+)?")


def pbs(*args):
    result = subprocess.run(args, check=False, text=True, capture_output=True, timeout=60)
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
    if "Exit_status" not in job:
        raise RuntimeError(f"PBS reported {job_id} finished without an exit status.")
    return int(job["Exit_status"])


def submit_attempt(root, start, end, attempts, path, state):
    ram = MEMORY_GB[len(attempts)]
    name = f"Pauli_{start}_{end}"
    output_dir = root / "results" / "shards" / f"{start}-{end}"
    log_path = root / "results" / "logs" / "auto-retry" / f"{name}_{ram}gb.log"
    variables = [f"PAULI_MIN_N={start}", f"PAULI_MAX_N={end}",
                 f"PAULI_OUTPUT_DIR={output_dir.relative_to(root)}"]
    variables.extend(variable for variable in ("MOSEKLM_LICENSE_FILE", "PAULI_VENV", "PAULI_MOSEK_THREADS")
                     if os.environ.get(variable))
    args = (
        "qsub", "-N", name, "-o", str(log_path),
        "-l", f"select=1:ncpus=12:mem={ram}gb", "-l", "walltime=08:00:00",
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
        raise RuntimeError(f"Unexpected job ID for {name}. Check PBS before resuming.")
    attempt["job_id"] = job_id
    save_state(path, state)
    print(f"{name}: submitted {job_id} at {ram} GB", flush=True)


def monitor(root, path, state):
    while True:
        active = False
        for start, end in RANGES:
            key = f"{start}-{end}"
            attempts = state[key]
            if attempts and attempts[-1]["exit_status"] is None:
                attempt = attempts[-1]
                code = finished_status(attempt["job_id"])
                if code is None:
                    active = True
                    continue
                attempt["exit_status"] = code
                save_state(path, state)
                print(f"{key}: {attempt['job_id']} exited with status {code}", flush=True)

            if attempts and (attempts[-1]["exit_status"] == 0 or len(attempts) == len(MEMORY_GB)):
                continue
            submit_attempt(root, start, end, attempts, path, state)
            active = True

        if not active:
            break
        time.sleep(POLL_SECONDS)

    failed = [key for key, attempts in state.items() if attempts[-1]["exit_status"] != 0]
    print(f"Completed {len(RANGES) - len(failed)} of {len(RANGES)} ranges.", flush=True)
    if failed:
        print(f"Failed at 64 GB: {', '.join(failed)}. See {path} and the attempt logs.", flush=True)
    return int(bool(failed))


def main():
    import fcntl

    root = Path.cwd()
    if not (root / "submit.pbs").is_file():
        raise RuntimeError("Run from the cluster's pauliEntropy checkout containing submit.pbs.")
    directory = root / "results" / "logs" / "auto-retry"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    with (directory / "controller.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A retry controller is already running for this checkout.") from error

        state = {f"{start}-{end}": [] for start, end in RANGES}
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
        return monitor(root, path, state)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
    except KeyboardInterrupt:
        sys.exit("Controller stopped. Restart it from the same checkout to resume monitoring.")
