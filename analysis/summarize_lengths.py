#!/usr/bin/env python3
"""Summarize the input/output token-length distribution of a throughput run.

Reads prompt_len_stats / output_len_stats from each <config>_ms<N>.json written by
tp_worker.py.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

import pandas as pd

STAT_COLS = ["n", "mean", "std", "min", "p50", "p90", "p99", "max"]


def describe(lengths):
    """Distribution summary matching tp_worker.len_stats / context_lengths._describe."""
    s = sorted(lengths)
    n = len(s)
    q = lambda p: s[min(n - 1, int(round(p * (n - 1))))]
    return {
        "n": n,
        "mean": round(st.mean(s), 1),
        "std": round(st.pstdev(s), 1),
        "min": s[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": s[-1],
    }


def load_points(run_dir):
    """Return [(cfg, max_seqs, json_dict), ...] for every result file in the run."""
    points = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*_ms*.json"))):
        m = re.match(r"(.+)_ms(\d+)$", os.path.basename(p)[:-5])
        if not m:
            continue
        with open(p) as f:
            points.append((m.group(1), int(m.group(2)), json.load(f)))
    return points


def _row(metric, stats, clip=None):
    r = {"metric": metric}
    for c in STAT_COLS:
        r[c] = stats.get(c)
    r["clip%"] = f"{100 * clip:.1f}" if clip is not None else "-"
    return r


def build_table(points):
    rows = []

    # input: dataset-fixed -> one representative row (first point that has it).
    in_stats = next((d.get("prompt_len_stats") for _, _, d in points if d.get("prompt_len_stats")), None)
    if in_stats:
        rows.append(_row("input", in_stats))

    # output: one row per config (first point seen for that config).
    seen = set()
    for cfg, _, d in points:
        if cfg in seen:
            continue
        os_stats = d.get("output_len_stats")
        if os_stats:
            seen.add(cfg)
            rows.append(_row(f"output ({cfg})", os_stats, clip=os_stats.get("clipped_frac")))

    # context (in+out): needs paired raw arrays (only present with --save-lengths).
    seen = set()
    for cfg, _, d in points:
        if cfg in seen:
            continue
        pl, ol = d.get("prompt_lens"), d.get("output_lens")
        if pl and ol:
            seen.add(cfg)
            ctx = [a + b for a, b in zip(pl, ol)]
            rows.append(_row(f"context ({cfg})", describe(ctx)))

    return pd.DataFrame(rows, columns=["metric"] + STAT_COLS + ["clip%"])


def plot_histograms(points, prefix):
    """Write <prefix>_input.png and <prefix>_output.png from the raw arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_arr = next((d.get("prompt_lens") for _, _, d in points if d.get("prompt_lens")), None)
    out_by_cfg = {}
    for cfg, _, d in points:
        if cfg not in out_by_cfg and d.get("output_lens"):
            out_by_cfg[cfg] = d["output_lens"]

    if not in_arr and not out_by_cfg:
        print("--plot: no raw length arrays found (run with --save-lengths)", file=sys.stderr)
        return []

    written = []
    if in_arr:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(in_arr, bins=50, color="#4C72B0")
        ax.set(title="input (prompt) token lengths", xlabel="tokens", ylabel="requests")
        fig.tight_layout()
        path = f"{prefix}_input.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    if out_by_cfg:
        fig, ax = plt.subplots(figsize=(7, 4))
        for cfg, arr in out_by_cfg.items():
            ax.hist(arr, bins=50, alpha=0.55, label=cfg)
        ax.set(title="output token lengths", xlabel="tokens", ylabel="requests")
        ax.legend()
        fig.tight_layout()
        path = f"{prefix}_output.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--fmt", choices=["table", "markdown", "csv"], default="table")
    ap.add_argument("--plot", metavar="PREFIX", default=None,
                    help="write <PREFIX>_input.png / <PREFIX>_output.png (needs --save-lengths data)")
    args = ap.parse_args()

    points = load_points(args.run_dir)
    if not points:
        sys.exit(f"no <config>_ms<N>.json files found in {args.run_dir}")

    out = build_table(points)
    if out.empty:
        sys.exit("no length stats in results (run tp_worker.py that writes prompt_len_stats)")

    if args.fmt == "csv":
        print(out.to_csv(index=False), end="")
    else:
        print(f"# run: {os.path.abspath(args.run_dir)}")
        print("# length profile (tokens/request) | clip% = share of natural outputs "
              "hitting the output_len cap\n")
        if args.fmt == "markdown":
            print(out.to_markdown(index=False))
        else:
            print(out.to_string(index=False))

    if args.plot:
        for path in plot_histograms(points, args.plot):
            print(f"# wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
