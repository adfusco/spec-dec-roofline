"""Split each engine step into target forward, draft phase, and overhead.

Reads the CUDA-event CSVs written by bench/draft_timer.py, keeps only steady-state
decode steps, and compares the measured verify/AR forward ratio against the model.
Run ids are pinned because later runs land in the same directories.
"""
import csv, glob, json, os, sys
import numpy as np
from predict_phase1 import t_w, t_c, t_wd, t_cd, w, w_d

V = D = 3
RES = "../results"


def load(run, cfg, B):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, f"{cfg}_ms{B}_draft_pid*.csv"))):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        return None
    if "target_fwd_ms" not in rows[0]:
        return "OLD"          # pre target-forward-bracket schema
    a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    keep = a["sched_tokens"] != 8192
    if cfg == "eagle3" and "num_query_tokens" in a:
        keep &= a["num_query_tokens"] != 8192
    a = {k: v[keep] for k, v in a.items()}
    m = ((a["sched_reqs"] == B) & (a["sched_tokens"] == B) if cfg == "baseline"
         else (a["batch"] == B) & (a["num_query_tokens"] == B * (V + 1)))
    return {k: v[m] for k, v in a.items()} if m.sum() else None


def latest(pat):
    d = sorted(glob.glob(pat))
    return d[-1] if d else None


# Pinned run ids, not "latest": later runs land in the same directories (e.g.
# 512_out128/20260902_204146 is a past-B_be sweep with no B=32) and silently
# replaced these, emptying the 512 row and dropping a point from the w_d fit.
CELLS = [("512", f"{RES}/govreport_llama_512_out128/20260901_190928", [32]),
         ("1k",  f"{RES}/govreport_llama_1k_out128/20260901_191513",  [16, 32, 64, 128]),
         ("2k",  f"{RES}/govreport_llama_2k_out128/20260901_185819",  [32])]

hdr = (f"{'tier':>5}{'B':>5}{'L':>6} | {'AR step':>8}{'fwd':>7}{'ovh':>7} | "
       f"{'SP step':>8}{'fwd':>7}{'draft':>7}{'ovh':>7} | "
       f"{'fwd ratio':>10}{'model':>7} | {'ovh ratio':>10}{'nfwd':>5}")
print(hdr); print("-" * len(hdr))
rows_out = []
for tier, pat, batches in CELLS:
    run = latest(pat)
    if run is None:
        continue
    for B in batches:
        ar, sp = load(run, "baseline", B), load(run, "eagle3", B)
        if ar == "OLD" or sp == "OLD":
            print(f"{tier:>5}{B:>5}   (old schema at {os.path.basename(run)} "
                  f"-- newer run not downloaded)")
            continue
        if ar is None or sp is None:
            print(f"{tier:>5}{B:>5}   (missing: "
                  f"{'baseline' if ar is None else ''}{'eagle3' if sp is None else ''})")
            continue
        med = lambda d, k: float(np.median(d[k]))
        ar_step, ar_fwd = med(ar, "step_ms"), med(ar, "target_fwd_ms")
        sp_step, sp_fwd = med(sp, "step_ms"), med(sp, "target_fwd_ms")
        dr = med(sp, "draft_ms")
        nfwd = int(np.median(sp["n_target_fwd"]))
        L = med(sp, "sum_seq_len") / B
        ar_ovh, sp_ovh = ar_step - ar_fwd, sp_step - sp_fwd - dr
        m_ar = (max(B * t_c, t_w) + B * w * (L + 1)) * 1e3
        m_vf = (max((V + 1) * B * t_c, t_w) + B * w * (L + V + 1)) * 1e3
        print(f"{tier:>5}{B:>5}{L:>6.0f} | {ar_step:>8.2f}{ar_fwd:>7.2f}{ar_ovh:>7.2f} | "
              f"{sp_step:>8.2f}{sp_fwd:>7.2f}{dr:>7.2f}{sp_ovh:>7.2f} | "
              f"{sp_fwd/ar_fwd:>10.2f}{m_vf/m_ar:>7.2f} | {sp_ovh/ar_ovh:>10.2f}{nfwd:>5}")
        rows_out.append((tier, B, L, dr, sp_fwd, sp_ovh, ar_fwd, ar_ovh))

fix = [r for r in rows_out if r[1] == 32]
if len(fix) >= 3:
    L = np.array([r[2] for r in fix]); y = np.array([r[3] / D / 1e3 for r in fix])
    A = np.vstack([32 * L, np.ones_like(L)]).T
    (sl, ic), *_ = np.linalg.lstsq(A, y, rcond=None)
    p = A @ np.array([sl, ic])
    r2 = 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    print(f"\n=== w_d at fixed B=32, L = {', '.join(f'{x:.0f}' for x in L)} "
          f"({L.max()/L.min():.1f}x) ===")
    print(f"  draft/pass  {', '.join(f'{v*1e3:.3f}' for v in y)} ms")
    print(f"  fitted w_d  {sl*1e9:.3f} ns   intercept {ic*1e3:.3f} ms   R^2 {r2:.3f}"
          f"   (dof = {len(fix)-2})")
    print(f"  model  w_d  {w_d*1e9:.3f} ns   t_wd      {t_wd*1e3:.3f} ms"
          f"   -> {sl/w_d:.1f}x")
