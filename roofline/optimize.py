"""Search serving configurations for the throughput- or speedup-optimal point.

Grid: hardware x dtype x target x TP x chain depth x batch, batch capped by B_KV.
Time primitives and S_e2e come from model.py. Acceptance is either fitted to the
measured runs or pinned with --acceptance.

The model is optimistic (it prices a verify pass as free whenever the weight load
dominates), so read the ranking rather than the absolute speedup.
"""
import argparse
import itertools
import math

import numpy as np

from model import (GiB, HW, Model, B_kv, S_e2e as _S_e2e, chain,
                   ridge_batch)

TARGETS = [
    # name              N        n_layers n_q n_kv d_h  hidden inter   ctx    draft_bytes    head repo
    ("Llama-3.1-8B",    8.03e9,  32, 32,  8, 128,  4096, 14336, 65536, 849_765_096,
     "TanBaby/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K"),
    ("Qwen3-8B",        8.2e9,   36, 32,  8, 128,  4096, 12288, 32768, 2_044_116_968,
     "RedHatAI/Qwen3-8B-speculator.eagle3"),
    ("Qwen3-14B",       14.8e9,  40, 40,  8, 128,  5120, 17408, 32768, 2_775_244_776,
     "RedHatAI/Qwen3-14B-speculator.eagle3"),
    ("Qwen3-32B",       32.8e9,  64, 64,  8, 128,  5120, 25600, 32768, 3_121_274_856,
     "RedHatAI/Qwen3-32B-speculator.eagle3"),
    ("Llama-3.3-70B",   70.6e9,  80, 64,  8, 128,  8192, 28672, 65536, 3_152_458_398,
     "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"),
]

FP8_C = {"H100-80": 1979e12, "H200": 1979e12, "L40S": 362e12,
         "A100-40": None, "A100-80": None}

# (bytes/weight, bytes/KV element).  kv fp8 halves kappa, which halves w and so
# DOUBLES r = t_c/w -- pushing L_crit up and speculation's break-even down.
DTYPES = {
    "bf16":   dict(bw=2, b_kv=2, fp8_compute=False),
    "fp8":    dict(bw=1, b_kv=2, fp8_compute=True),
    "fp8kv":  dict(bw=1, b_kv=1, fp8_compute=True),
}

TP_CHOICES = (1, 2, 4, 8)
# Capped at 2048: a DB column longer than that is a document, not a field.
L_IN = (256, 512, 896, 1024, 2048)   # 896 ~ PDMX (887)
# Capped at 1024: nothing a DB column plausibly holds is longer.  Measured
# in-DB outputs are far shorter still -- PDMX 71, relqueries 9.
L_OUT = (32, 64, 128, 192, 256, 384, 512, 1024)
BATCHES = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
DEPTHS = (1, 2, 3, 4, 5, 7)       # num_speculative_tokens
# Multi-GPU TP without NVLink (L40S is PCIe) pays all-reduce cost per layer that
# this model does not price at all -- S(B,L) is TP-invariant here by assumption.
NVLINK = {"A100-40", "A100-80", "H100-80", "H200"}
H_PREFIX = 0.0                    # GovReport measured 0.0-0.8%; assume none


def eagle3_bytes(hidden, inter, n_q, n_kv, d_h, draft_vocab=32000, b=2):
    """One EAGLE-3 decoder layer + the 3h->h fusion + a reduced-vocab LM head.

    Validated against the weighed Llama-3.1-8B head: predicts 799 MB vs 850 MB
    actual (the remainder is norms and buffers), so it is a mild UNDER-estimate,
    i.e. optimistic for speculation.
    """
    attn = hidden * n_q * d_h * 2 + hidden * n_kv * d_h * 2
    mlp = 3 * hidden * inter
    fusion = 3 * hidden * hidden
    lm_head = draft_vocab * hidden
    return (attn + mlp + fusion + lm_head) * b


A_MEASURED = ((30.0, 0.613), (125.0, 0.660), (470.0, 0.733))


ACCEPT_OVERRIDE = None      # set by --acceptance; None = use the fit


def accept_rate(L_out):
    """Per-position acceptance, log-fitted to the measured runs and clamped to their
    range, or the scalar set by --acceptance."""
    if ACCEPT_OVERRIDE is not None:
        return ACCEPT_OVERRIDE
    x = np.log([p[0] for p in A_MEASURED])
    y = np.array([p[1] for p in A_MEASURED])
    d, c = np.polyfit(x, y, 1)
    lo, hi = A_MEASURED[0][0], A_MEASURED[-1][0]
    return float(c + d * math.log(min(max(L_out, lo), hi)))


def attn_coeff(n_layers, n_q, d_h, C):
    """DEPRECATED shim: `a` now lives on Model as `m.a` / `m.a_d`.

    Kept only so callers that computed it externally keep working; the value is
    identical to Model's.  New code should read m.a.
    """
    return 4.0 * n_layers * n_q * d_h / C


def s_e2e(m, B, L_in, L_out, D, V, Om, h=H_PREFIX, a=None, a_d=None):
    """End-to-end speedup, prefill plus decode. Returns (S, prefill share, AR time)."""
    S, pre = _S_e2e(B, L_in, L_out, D, V, Om, m, h)
    L = L_in + L_out / 2.0
    from model import T_ar, T_pre_ar
    tot_ar = T_pre_ar(B, L_in, m, h) + L_out * T_ar(B, L, m)
    return S, pre, tot_ar


def throughput(m, B, L_in, L_out, tp, tot_ar, S):
    """Output tokens/s per GPU. model.py primitives are one-GPU-equivalent, so per-GPU
    throughput is B*L_out/tot_ar with no tp factor; dividing by tp double-counts."""
    ar = B * L_out / tot_ar
    return ar, ar * S


def apply_dtype(hw_key, dt):
    """Return (hw dict with C/bw overridden, b_kv, weight_only flag)."""
    d = DTYPES[dt]
    hw = dict(HW[hw_key])
    weight_only = False
    if d["fp8_compute"]:
        c8 = FP8_C.get(hw_key)
        if c8 is None:
            weight_only = True          # no FP8 unit: bytes shrink, C does not
        else:
            hw["C"] = c8
    hw["bw"] = d["bw"]
    return hw, d["b_kv"], weight_only


def search(min_batch, max_batch, hw_keys, max_depth=5, max_l_out=512,
           nvlink_only=True, batch_mode="sweep", dtypes=("bf16",)):
    rows = []
    for hw_key, dt in itertools.product(hw_keys, dtypes):
        hw, b_kv_bytes, weight_only = apply_dtype(hw_key, dt)
        for (name, N, nl, nq, nkv, dh, hid, inter, ctx, dbytes, repo) in TARGETS:
            # dbytes is the head's bf16 checkpoint.  vLLM quantizes only the
            # target, so the head stays bf16 even when the target is fp8.
            db = dbytes or eagle3_bytes(hid, inter, nq, nkv, dh)
            for tp in TP_CHOICES:
                # TP>1 without NVLink (L40S is PCIe) pays a per-layer all-reduce
                # this model prices at zero -- S(B,L) is TP-invariant here by
                # assumption, so those points would be fiction.
                if tp > 1 and nvlink_only and hw_key not in NVLINK:
                    continue
                # weights must fit with headroom for KV
                if tp * (hw["HBM_GIB"] - hw["RES_GIB"]) * GiB <= (N * hw["bw"] + db) * 1.15:
                    continue
                m = Model(name, N, nl, nq, nkv, dh, hw, b_kv=b_kv_bytes,
                          draft_bytes=db, draft_bw=2, draft_C=HW[hw_key]["C"])
                # NB: `a` is already the acceptance rate further down this
                # loop, so the attention coefficients must not use that name.
                attn_a = attn_coeff(nl, nq, dh, hw["C"])
                attn_ad = attn_coeff(1, nq, dh, hw["C"])   # 1-layer EAGLE head
                if tp < m.min_tp:
                    continue
                for L_in, L_out in itertools.product(L_IN, L_OUT):
                    if L_in + L_out > ctx or L_out > max_l_out:
                        continue
                    a = accept_rate(L_out)
                    for g in (d for d in DEPTHS if d <= max_depth):
                        D, V, Om = chain(g, a)
                        cap = B_kv(L_in + L_out, m, tp)
                        if not np.isfinite(cap) or cap < min_batch:
                            continue
                        if batch_mode == "bkv":
                            cand = [int(min(cap, max_batch))]
                        else:
                            cand = [b for b in BATCHES
                                    if min_batch <= b <= int(min(max_batch, cap))]
                        for B in cand:
                            if B < min_batch:
                                continue
                            S, pre, tot = s_e2e(m, B, L_in, L_out, D, V, Om,
                                                a=attn_a, a_d=attn_ad)
                            tp_ar, tp_sp = throughput(m, B, L_in, L_out, tp,
                                                      tot, S)
                            rows.append(dict(
                                tp_ar=tp_ar, tp_sp=tp_sp, dtok=tp_sp - tp_ar,
                                hw=hw_key, dtype=dt + ("*" if weight_only else ""),
                                model=name, repo=repo, tp=tp, B=B,
                                L_in=L_in, L_out=L_out, g=g, a=a, Om=Om, S=S,
                                pre=pre, r=m.r, Tstar=ridge_batch(hw),
                                B_kv=cap, draft_gb=db / 1e9))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-batch", type=int, default=32,
                    help="floor on batch: below this it is not an offline "
                         "serving point (default 32)")
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--hw", default="A100-40,A100-80,H100-80,H200,L40S",
                    help="comma-separated keys from model.py's HW table")
    ap.add_argument("--max-depth", type=int, default=5,
                    help="largest num_speculative_tokens to consider; EAGLE-3 "
                         "heads are trained shallow and depth is untested here")
    ap.add_argument("--max-l-out", type=int, default=512,
                    help="largest L_out; beyond the measured range acceptance "
                         "is held flat, so longer outputs only add prefill "
                         "dilution, but they are also untested")
    ap.add_argument("--allow-pcie-tp", action="store_true",
                    help="allow TP>1 on non-NVLink parts (L40S). Off by "
                         "default: the model prices TP comms at zero")
    ap.add_argument("--batch-mode", choices=("sweep", "bkv"), default="sweep",
                    help="'bkv' pins B to the KV-capacity ceiling, which is what "
                         "offline serving actually does to maximise throughput")
    ap.add_argument("--dtype", default="bf16",
                    help="comma-separated: bf16, fp8, fp8kv, or 'all'. fp8 on a "
                         "part with no FP8 unit (A100) is weight-only and is "
                         "flagged with * in the output")
    ap.add_argument("--acceptance", default="fit",
                    help="'fit' = log fit to the Phase 1 measurements, clamped "
                         "to [30,470] tokens; or a float in (0,1) applied "
                         "uniformly to every cell (e.g. 0.80 for an optimistic "
                         "sensitivity). Measured range was 0.613-0.733.")
    ap.add_argument("--objective", choices=("speedup", "gain", "tput"),
                    default="speedup",
                    help="what to maximise. 'speedup' = S_e2e, the ratio. "
                         "'gain' = extra output tok/s/GPU that speculation "
                         "buys, tput_AR*(S-1) -- the absolute quantity, which "
                         "trades ratio for a faster base. 'tput' = the "
                         "speculative throughput itself, which mostly just "
                         "picks the fastest part and ignores speculation.")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    global ACCEPT_OVERRIDE
    if args.acceptance != "fit":
        ACCEPT_OVERRIDE = float(args.acceptance)
        if not 0 < ACCEPT_OVERRIDE < 1:
            raise SystemExit("--acceptance must be 'fit' or in (0,1)")
    hw_keys = [k.strip() for k in args.hw.split(",") if k.strip() in HW]
    dts = (tuple(DTYPES) if args.dtype == "all"
           else tuple(d.strip() for d in args.dtype.split(",") if d.strip() in DTYPES))
    if not dts:
        raise SystemExit(f"--dtype must name some of {list(DTYPES)} or 'all'")
    rows = search(args.min_batch, args.max_batch, hw_keys,
                  max_depth=args.max_depth, max_l_out=args.max_l_out,
                  nvlink_only=not args.allow_pcie_tp,
                  batch_mode=args.batch_mode, dtypes=dts)
    if not rows:
        raise SystemExit("no configuration satisfied the constraints")
    obj = {"speedup": "S", "gain": "dtok", "tput": "tp_sp"}[args.objective]
    rows.sort(key=lambda r: -r[obj])

    print(f"searched {len(rows)} feasible points   "
          f"batch in [{args.min_batch}, {args.max_batch}]   hw={hw_keys}\n")
    hdr = (f"{'gain':>7}{'S_e2e':>7}{'AR t/s':>8}{'spec':>8}{'hw':>9}{'dtype':>7}"
           f"{'model':>14}{'TP':>3}{'B':>5}{'L_in':>6}{'L_out':>6}{'D':>3}"
           f"{'Omega':>6}{'pre%':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows[:args.top]:
        print(f"{r['dtok']:>7.0f}{r['S']:>6.2f}x{r['tp_ar']:>8.0f}{r['tp_sp']:>8.0f}"
              f"{r['hw']:>9}{r['dtype']:>7}{r['model']:>14}"
              f"{r['tp']:>3}{r['B']:>5}{r['L_in']:>6}{r['L_out']:>6}{r['g']:>3}"
              f"{r['Om']:>6.2f}{r['pre']:>5.0%}")

    w = rows[0]
    print(f"\n{'='*72}\nBEST REALISTIC CONFIGURATION"
          f"  (maximising {args.objective})\n{'='*72}")
    print(f"  hardware      {w['hw']} x TP={w['tp']}, dtype={w['dtype']}"
          f"   (T* = {w['Tstar']:.0f}, r = {w['r']:.0f})")
    print(f"  target        {w['model']}")
    print(f"  drafter       {w['repo']}  ({w['draft_gb']:.2f} GB)")
    print(f"  workload      L_in = {w['L_in']}, L_out = {w['L_out']}")
    print(f"  spec depth    num_speculative_tokens = {w['g']}  "
          f"(a = {w['a']:.3f} -> Omega = {w['Om']:.2f})")
    print(f"  batch         {w['B']}   (B_KV = {w['B_kv']:.0f})")
    print(f"  predicted     S_e2e = {w['S']:.2f}x, prefill = {w['pre']:.0%} of AR runtime")
    print(f"  throughput    {w['tp_ar']:.0f} -> {w['tp_sp']:.0f} out tok/s/GPU "
          f"(gain {w['dtok']:.0f}, x{w['tp']} GPU = {w['dtok']*w['tp']:.0f}/node)")

    # batch curve at the winning workload -- what the sweep should look like
    hw, b_kv_bytes, _ = apply_dtype(w["hw"], w["dtype"].rstrip("*"))
    spec = next(t for t in TARGETS if t[0] == w["model"])
    db = spec[9] or eagle3_bytes(spec[6], spec[7], spec[3], spec[4], spec[5])
    m = Model(spec[0], spec[1], spec[2], spec[3], spec[4], spec[5], hw,
              b_kv=b_kv_bytes, draft_bytes=db, draft_bw=2, draft_C=HW[w["hw"]]["C"])
    D, V, Om = chain(w["g"], w["a"])
    aw = attn_coeff(spec[2], spec[3], spec[5], hw["C"])
    aw_d = attn_coeff(1, spec[3], spec[5], hw["C"])
    print(f"\n  batch sweep at this workload (B_KV = {w['B_kv']:.0f}):")
    print(f"    {'B':>6}{'S_e2e':>9}{'AR t/s':>9}{'spec':>9}{'gain':>9}")
    for B in BATCHES:
        if B > w["B_kv"]:
            break
        S, _, tot = s_e2e(m, B, w["L_in"], w["L_out"], D, V, Om, a=aw, a_d=aw_d)
        t_ar, t_sp = throughput(m, B, w["L_in"], w["L_out"], w["tp"], tot, S)
        mark = "  <- proposed" if B == w["B"] else ""
        print(f"    {B:>6}{S:>8.2f}x{t_ar:>9.0f}{t_sp:>9.0f}{t_sp-t_ar:>9.0f}{mark}")
    print("\n  NOTE: the model over-predicts measured decode speedup by 31-62%")
    print("  (see README).  Treat S_e2e as a ranking, not a forecast.")


if __name__ == "__main__":
    main()
