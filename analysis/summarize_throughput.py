#!/usr/bin/env python3
"""Summarize an offline throughput run into one table.

A run is a directory of <config>_ms<batch>.json files written by tp_worker.py; this
derives tok/s, acceptance, MFU and steady-state throughput for each point.
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

# --- GPU peak dense bf16 tensor-core throughput (TFLOP/s per GPU), for MFU ---
GPU_TFLOPS = {
    "a100-40gb": 312.0,    # A100 (bf16, no sparsity)
    "a100-80gb": 312.0,
    "h100": 989.0,         # H100 SXM (bf16 tensor core; commonly-cited peak)
    "h100-nvl": 1979.0,
    "h200": 989.0,
}

DEFAULT_N_PARAMS_B = 32.8   # Qwen3-32B parameter count (billions)


def steady_stats(det, max_seqs, plateau_frac=0.9):
    """Throughput over the window where concurrency held at the cap, which trims
    ramp-up and drain."""
    fs, fin = det.get("first_scheduled"), det.get("finished")
    ft, ol = det.get("first_token"), det.get("output_lens")
    if not fs or not fin or len(fs) != len(fin):
        return {}
    t0 = min(fs)
    starts = [s - t0 for s in fs]
    ends = [e - t0 for e in fin]
    dur = max(ends)
    if dur <= 0:
        return {}
    ev = sorted([(s, 1) for s in starts] + [(e, -1) for e in ends], key=lambda x: (x[0], x[1]))
    cur = pk = 0
    for _, dl in ev:
        cur += dl
        pk = max(pk, cur)
    conc_mean = sum(e - s for s, e in zip(starts, ends)) / dur
    thresh = plateau_frac * (max_seqs or pk)
    cur, prev_t, w0, w1 = 0, 0.0, None, None
    for t, dl in ev:
        if cur >= thresh:
            if w0 is None:
                w0 = prev_t
            w1 = t
        cur += dl
        prev_t = t
    out = dict(conc_pk=float(pk), conc_mean=conc_mean, w0=w0, w1=w1)
    if w0 is not None and w1 is not None and (w1 - w0) >= 0.05 * dur and ol:
        tok = 0.0
        for i in range(len(fin)):
            a = (ft[i] - t0) if (ft and ft[i]) else starts[i]
            b = ends[i]
            if b > a and ol[i]:
                lo, hi = max(a, w0), min(b, w1)
                if hi > lo:
                    tok += ol[i] * (hi - lo) / (b - a)
        out["ss_out_tps"] = tok / (w1 - w0)
    return out


def _step_series(intervals):
    """Turn [(start, end), ...] active intervals into a (times, levels) step
    function: level jumps +1 at each start, -1 at each end (departures before
    arrivals on ties). Plot with step(where='post')."""
    ev = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals],
                key=lambda x: (x[0], x[1]))
    times, levels, cur = [0.0], [0], 0
    for t, dl in ev:
        cur += dl
        times.append(t)
        levels.append(cur)
    return times, levels


def plot_concurrency(run_dir, prefix, plateau_frac=0.9):
    """Write one concurrency-over-time PNG per detailed result: total in-flight
    plus the prefilling/decoding split, with the max_num_seqs cap and the
    steady-state window overlaid. Needs runs made with `tp_worker.py
    --save-detailed` (else the JSON has no `detailed` block)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*_ms*.json"))):
        # Skip step-log sidecars (baseline_ms8_step_stats.json matches *_ms*.json).
        if path.endswith("_step_stats.json"):
            continue
        with open(path) as f:
            d = json.load(f)
        det = d.get("detailed")
        if not det or not det.get("first_scheduled") or not det.get("finished"):
            continue
        fs, fin = det["first_scheduled"], det["finished"]
        ft = det.get("first_token") or [None] * len(fs)
        t0 = min(fs)
        starts = [s - t0 for s in fs]
        ends = [e - t0 for e in fin]
        ftr = [(x - t0) if x else None for x in ft]
        # Each active request is prefilling over [scheduled, first_token] then
        # decoding over [first_token, finished]; the two sum to the total batch.
        total = list(zip(starts, ends))
        prefill = [(s, f if f is not None else s) for s, f in zip(starts, ftr)]
        decode = [(f if f is not None else s, e) for f, s, e in zip(ftr, starts, ends)]

        name = os.path.basename(path)[:-5]
        max_seqs = d.get("max_num_seqs")
        st = steady_stats(det, max_seqs or 0, plateau_frac)

        fig, ax = plt.subplots(figsize=(9, 4))
        for label, iv in (("total", total), ("decoding", decode), ("prefilling", prefill)):
            t, lv = _step_series(iv)
            ax.step(t, lv, where="post", lw=1.4, label=label)
        if max_seqs:
            ax.axhline(max_seqs, ls="--", color="gray", lw=1, label=f"max_num_seqs={max_seqs}")
        if st.get("w0") is not None and st.get("w1") is not None:
            ax.axvspan(st["w0"], st["w1"], color="green", alpha=0.08, label="steady window")
        ax.set(title=f"concurrency over time — {name}", xlabel="time (s)",
               ylabel="in-flight requests")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        outpath = f"{prefix}_{name}.png"
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        written.append(outpath)

    if not written:
        print("--plot-conc: no result had a 'detailed' block (run with --save-detailed)",
              file=sys.stderr)
    return written


def load_row(path):
    with open(path) as f:
        d = json.load(f)
    m = re.match(r"(.+)_ms(\d+)$", os.path.basename(path)[:-5])
    if not m:
        raise ValueError("filename not <config>_ms<N>.json")
    cfg = m.group(1)
    B = int(m.group(2))
    elapsed = d["elapsed_time"]
    n = d.get("num_prompts") or 0
    # Marginal per-position acceptance a_i -> conditional a_i/a_(i-1).
    marg = d.get("per_pos_acceptance") or []
    cond = [marg[i] / (marg[i - 1] if i else 1.0) if (i == 0 or marg[i - 1]) else 0.0
            for i in range(len(marg))]
    det = steady_stats(d["detailed"], d.get("max_num_seqs", B)) if d.get("detailed") else {}
    return dict(
        cfg=cfg,
        max_seqs=d.get("max_num_seqs", B),
        n=n,
        elapsed=elapsed,
        req_s=d.get("requests_per_second", n / elapsed if elapsed else float("nan")),
        out_tps=d.get("output_tokens_per_second",
                      d.get("total_output_tokens", 0) / elapsed),
        tot_tps=d.get("total_tokens_per_second", d["total_num_tokens"] / elapsed),
        tot_tokens=d["total_num_tokens"],
        avg_out_len=d.get("avg_output_len",
                          (d.get("total_output_tokens", 0) / n) if n else float("nan")),
        acc_len=d.get("acceptance_length"),
        cond="/".join(f"{c:.2f}" for c in cond) or "-",
        ss_out_tps=det.get("ss_out_tps"),
        conc_pk=det.get("conc_pk"),
    )


def build_table(run_dir, peak_flops_agg, n_params):
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*_ms*.json"))):
        if p.endswith("_step_stats.json"):
            continue
        try:
            rows.append(load_row(p))
        except Exception as e:                            # skip aux JSONs, keep going
            print(f"skip {os.path.basename(p)}: {e}", file=sys.stderr)
    if not rows:
        sys.exit(f"no <config>_ms<N>.json files found in {run_dir}")

    base_out = {r["max_seqs"]: r["out_tps"] for r in rows if r["cfg"] == "baseline"}
    base_ss = {r["max_seqs"]: r["ss_out_tps"] for r in rows
               if r["cfg"] == "baseline" and r.get("ss_out_tps") is not None}
    for r in rows:
        r["spd"] = (r["out_tps"] / base_out[r["max_seqs"]]
                    if r["max_seqs"] in base_out and base_out[r["max_seqs"]] else float("nan"))
        # Steady-state speedup: same pairing as speedup, but on drain-trimmed rates.
        ss, ss_b = r.get("ss_out_tps"), base_ss.get(r["max_seqs"])
        r["ss_spd"] = (ss / ss_b) if (ss is not None and ss_b) else float("nan")
        # forward-only MFU: 2 FLOPs/param/token over all processed tokens.
        r["mfu"] = 100 * (2 * n_params * r["tot_tokens"] / r["elapsed"]) / peak_flops_agg

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["cfg", "max_seqs"], key=lambda c: c.map(
        lambda v: (v != "baseline", v)) if c.name == "cfg" else c)
    out = pd.DataFrame({
        "cfg": df["cfg"],
        "max_seqs": df["max_seqs"],
        "req/s": df["req_s"].round(2),
        "out_tok/s": df["out_tps"].round(0).astype(int),
        "tot_tok/s": df["tot_tps"].round(0).astype(int),
        "speedup": df["spd"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "-"),
        "avg_out_len": df["avg_out_len"].round(0).astype("Int64"),
        "acc_len": df["acc_len"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        "per_pos_accept": df["cond"],
        "MFU%": df["mfu"].round(1),
    })
    # Optional: only show steady-state / true-concurrency columns for runs made
    # with --save-detailed (else the whole column would be blank).
    if df["ss_out_tps"].notna().any():
        out["ss_out_tok/s"] = df["ss_out_tps"].round(0).astype("Int64")
        out["ss_speedup"] = df["ss_spd"].map(
            lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
    if df["conc_pk"].notna().any():
        out["conc_pk"] = df["conc_pk"].round(0).astype("Int64")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--fmt", choices=["table", "markdown", "csv"], default="table")
    ap.add_argument("--gpu", choices=sorted(GPU_TFLOPS), default="a100-40gb",
                    help="GPU preset for peak dense bf16 TFLOP/s (MFU denominator)")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel size")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="override peak TFLOP/s per GPU (else from --gpu)")
    ap.add_argument("--n-params-b", type=float, default=DEFAULT_N_PARAMS_B,
                    help="target model parameter count in billions (for MFU)")
    ap.add_argument("--plot-conc", metavar="PREFIX", default=None,
                    help="write <PREFIX>_<cfg>_ms<N>.png concurrency-over-time plots "
                         "(needs --save-detailed data)")
    args = ap.parse_args()

    peak_per_gpu = args.peak_tflops if args.peak_tflops is not None else GPU_TFLOPS[args.gpu]
    peak_flops_agg = args.tp * peak_per_gpu * 1e12
    n_params = args.n_params_b * 1e9

    if args.plot_conc:
        for p in plot_concurrency(args.run_dir, args.plot_conc):
            print(f"# wrote {p}", file=sys.stderr)

    out = build_table(args.run_dir, peak_flops_agg, n_params)
    if args.fmt == "csv":
        print(out.to_csv(index=False), end="")
        return

    print(f"# run: {os.path.abspath(args.run_dir)}")
    print(f"# offline throughput | GPU={args.gpu} x TP={args.tp} | "
          f"peak={args.tp * peak_per_gpu:.0f} TFLOP/s (dense bf16) | "
          f"N_params={args.n_params_b:.1f}B")
    print(f"# out_tok/s = decode throughput; tot_tok/s = (prompt+output)/elapsed; "
          f"avg_out_len = mean output tokens/req; acc_len = mean accepted+bonus/step; "
          f"per_pos_accept = conditional a_i/a_(i-1); "
          f"MFU% = forward-only, useful-token model FLOPs utilization\n")
    if args.fmt == "markdown":
        print(out.to_markdown(index=False))
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
