# Pauli entropy: Colab -> GitHub -> cluster -> GitHub

1. Edit/run `pauli_combined_entropy.ipynb` in Colab. Add `GITHUB_TOKEN` to
   Colab Secrets, grant the notebook access, and give the token Contents: write
   permission for `CatUwUKitty/pauliEntropy`. Create the repository first.
2. Run the final exporter cell. It saves the current notebook (including outputs),
   `run_cluster.py`, `submit.pbs`, `requirements.txt`, `push_results.py`, this guide,
   and existing CSV/PNG results under `pauli_entropy_results/`. Existing GitHub
   result files are preserved; set OVERWRITE_GITHUB_RESULTS=True in the exporter
   only when you intend to replace them with the Colab versions.
   You can run the exporter alone without repeating the sweeps; it uses notebook
   source settings. If needed, set EXPORT_RESULTS_DIR to your saved results folder.
3. On the cluster, clone once or update your existing checkout:

   ```bash
   git clone --branch main https://github.com/CatUwUKitty/pauliEntropy.git
   cd pauliEntropy
   # For later exports, from this checkout:
   git pull --ff-only origin main
   ```

4. Review `submit.pbs`: project/queue/resources, Python module, virtual environment,
   and MOSEK license path are copied from the Werner setup. Create the virtual
   environment once if necessary (`python3 -m venv ~/myenv`, using the cluster's
   Python module). Dependencies are installed by the PBS job. Keep the MOSEK
   license on the cluster, outside this repository.

   ```bash
   qsub submit.pbs
   # Or, with dependencies installed in your active environment:
   python run_cluster.py --p 0.05 --collision-epsilon 0.001 --n 1 2 3 --solver CLARABEL
   python run_cluster.py --help
   ```

   The runner uses the notebook's saved parameters unless overridden. Changing
   `--collision-epsilon` also changes the min-entropy squared radius; use
   `--min-entropy-epsilon` for an independent override. It excludes Colab setup,
   plots, baseline-validation runs, and the exporter. Existing successful CSV
   checkpoints are reused when RESUME is enabled in the notebook or via `--resume`.
   Failed points are retried; a run with errors exits nonzero.

5. After the job completes, push the data from the cluster checkout:

   ```bash
   git config user.name CatUwUKitty
   git config user.email CatUwUKitty@users.noreply.github.com
   python push_results.py
   # If you chose a custom output folder inside this checkout:
   python push_results.py --results-dir my_results
   ```

   Configure Git authentication on the cluster separately (SSH or a credential
   helper). The exporter does not save the Colab token. The upload helper stages
   only CSV/PNG results and requires other staged changes to be resolved first.
   If Git rejects a push because the remote advanced, reconcile it with
   `git pull --rebase origin main`, then rerun the upload helper.
   Do not push results while a job is still updating checkpoint files.
   Pull the repository again to bring the results back to your other machines.

The exporter generates files before contacting GitHub and never submits a job.
Set `PUSH_TO_GITHUB = False` to export locally. Solver settings and numerical
formulations remain in the notebook; edit them there and export again.
