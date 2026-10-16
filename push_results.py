#!/usr/bin/env python3
"""Run from the cluster checkout after the job: python push_results.py."""
import argparse
import subprocess
from pathlib import Path

BRANCH = 'main'

def git(*args, capture=False):
    result = subprocess.run(['git', *args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None

def main():
    parser = argparse.ArgumentParser(description='Commit and push Pauli CSV/PNG results.')
    parser.add_argument('--results-dir', default='pauli_entropy_results')
    parser.add_argument('--message', default='Update Pauli entropy results from cluster')
    args = parser.parse_args()
    root = Path(git('rev-parse', '--show-toplevel', capture=True)).resolve()
    directory = Path(args.results_dir).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise SystemExit('Choose a results folder inside this Git checkout.')
    if git('branch', '--show-current', capture=True) != BRANCH:
        raise SystemExit(f'Check out {BRANCH} before pushing results.')
    if git('diff', '--cached', '--name-only', capture=True):
        raise SystemExit('Commit or unstage existing staged changes before pushing results.')
    files = sorted(path.relative_to(root).as_posix() for path in directory.rglob('*')
                   if path.is_file() and path.suffix.lower() in ('.csv', '.png')
                   and 'cluster_export' not in path.relative_to(directory).parts
                   and path.resolve().is_relative_to(root))
    if not files:
        raise SystemExit(f'No CSV/PNG results found in {directory}.')
    # Small batches avoid command-line limits for large sweeps.
    for offset in range(0, len(files), 100):
        git('-C', str(root), 'add', '--', *files[offset:offset + 100])
    if git('diff', '--cached', '--name-only', capture=True):
        git('commit', '-m', args.message)
    git('push', 'origin', BRANCH)

if __name__ == '__main__':
    main()
