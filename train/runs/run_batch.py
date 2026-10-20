#!/usr/bin/env python3
"""
run_batch.py — fire up multiple AL runs (seed / strategy combos) from a runs file.

A runs file (e.g. train/runs/k562.txt, written by make_runs.py) holds one full
`python train/al_train_*.py ...` command per line. This script runs those lines
with a configurable number of parallel workers, optional strategy/seed
filtering, per-run logging, and a dry-run mode. Run it from the package root.

Examples:
    python train/runs/run_batch.py                      # all lines in train/runs/k562.txt, one at a time
    python train/runs/run_batch.py -j 4                 # 4 runs in parallel
    python train/runs/run_batch.py -s tntk,ntk -S 0-9  # only those arms, seeds 0..9
    python train/runs/run_batch.py -f train/runs/bmdm.txt -j 8
    python train/runs/run_batch.py -n                   # dry run: print commands, run nothing

Tip: everything you'd commonly tweak lives in the DEFAULTS block right below.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# DEFAULTS — edit these freely; each is also overridable from the command line.
# ---------------------------------------------------------------------------
RUNFILE = "train/runs/k562.txt"            # file of one command per line
JOBS = 1                                   # number of runs in parallel
PYTHON = os.environ.get("PYTHON", sys.executable)  # interpreter to use
STRATEGIES = None                          # e.g. ["tntk", "ntk"]; None = all
SEEDS = None                               # e.g. [0, 1, 2] or range(0, 10); None = all
LOGDIR = None                              # None -> logs/<runfile-stem>
DRYRUN = False
# ---------------------------------------------------------------------------

STRAT_RE = re.compile(r"strategy[ =]+([A-Za-z_]+)")
SEED_RE = re.compile(r"init-seed[ =]+(\d+)")


def parse_int_set(spec):
    """Expand a spec like '0-9,42' into a set of ints. Returns None if spec is None/empty."""
    if not spec:
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def parse_str_set(spec):
    """Normalize a strategy spec (list or 'a,b' string) into a set. None -> None."""
    if not spec:
        return None
    if isinstance(spec, (list, tuple, set)):
        return set(spec)
    return {s.strip() for s in str(spec).split(",") if s.strip()}


def field(regex, line):
    m = regex.search(line)
    return m.group(1) if m else None


def load_jobs(runfile, python, strat_set, seed_set):
    """Read the runs file, apply filters, return a list of (label, command) tuples."""
    jobs = []
    total = 0
    with open(runfile) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            total += 1
            strat = field(STRAT_RE, line)
            seed = field(SEED_RE, line)
            if strat_set is not None and strat not in strat_set:
                continue
            if seed_set is not None and (seed is None or int(seed) not in seed_set):
                continue
            # swap the leading `python` for the chosen interpreter
            cmd = re.sub(r"^python\b", python, line)
            label = f"{strat or 'NA'}_seed{seed or 'NA'}"
            jobs.append((label, cmd))
    return jobs, total


def run_one(idx, n, label, cmd, logdir):
    """Run a single command, streaming its output to logs/<label>.log."""
    log_path = os.path.join(logdir, f"{label}.log")
    start = time.time()
    print(f"[start {idx}/{n}] {label}  -> {log_path}", flush=True)
    with open(log_path, "w") as log:
        log.write(f"$ {cmd}\n\n")
        log.flush()
        rc = subprocess.call(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - start
    status = "ok  " if rc == 0 else f"FAIL(rc={rc})"
    print(f"[ {status} {idx}/{n}] {label}  ({dt:.0f}s)", flush=True)
    return label, rc


def main():
    p = argparse.ArgumentParser(description="Run seed/strategy AL combos from a runs file.")
    p.add_argument("-f", "--runfile", default=RUNFILE, help=f"runs file (default: {RUNFILE})")
    p.add_argument("-j", "--jobs", type=int, default=JOBS, help=f"parallel runs (default: {JOBS})")
    p.add_argument("-s", "--strategies", default=STRATEGIES,
                   help="comma list, e.g. tntk,ntk (default: all)")
    p.add_argument("-S", "--seeds", default=SEEDS,
                   help="comma list and/or ranges, e.g. 0-9,42 (default: all)")
    p.add_argument("-l", "--logdir", default=LOGDIR, help="log directory (default: logs/<runfile-stem>)")
    p.add_argument("-p", "--python", default=PYTHON, help=f"interpreter (default: {PYTHON})")
    p.add_argument("-n", "--dry-run", action="store_true", default=DRYRUN, help="print commands, run nothing")
    args = p.parse_args()

    if not os.path.isfile(args.runfile):
        sys.exit(f"error: runs file not found: {args.runfile}")

    strat_set = parse_str_set(args.strategies)
    seed_set = parse_int_set(args.seeds) if not isinstance(args.seeds, (list, range)) else set(args.seeds)

    jobs, total = load_jobs(args.runfile, args.python, strat_set, seed_set)
    logdir = args.logdir or os.path.join("logs", os.path.splitext(os.path.basename(args.runfile))[0])

    print(f"runs file  : {args.runfile}")
    print(f"interpreter: {args.python}")
    print(f"selected   : {len(jobs)} / {total} commands")
    print(f"parallel   : {args.jobs}    logs: {logdir}")

    if not jobs:
        print("nothing to run.")
        return

    if args.dry_run:
        print("---- dry run (commands not executed) ----")
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"[{i:3d}] {cmd}")
        return

    if not (os.path.isfile(args.python) or os.path.isabs(args.python) is False):
        # light sanity check; a bare name like "python" is allowed (resolved via PATH)
        pass
    os.makedirs(logdir, exist_ok=True)

    n = len(jobs)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(run_one, i, n, label, cmd, logdir): label
            for i, (label, cmd) in enumerate(jobs, 1)
        }
        for fut in as_completed(futs):
            label, rc = fut.result()
            if rc != 0:
                failures.append(label)

    print(f"\nall done. {n - len(failures)}/{n} succeeded. logs in {logdir}/")
    if failures:
        print("failed runs: " + ", ".join(sorted(failures)))
        sys.exit(1)


if __name__ == "__main__":
    main()
