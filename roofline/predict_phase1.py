"""Time primitives for Llama-3.1-8B + EAGLE-3 on A100-40.

t_w, t_c, w and the drafter equivalents, derived from hardware and architecture
constants. Imported by decomp3.py to compare measured steps against the model.
"""
import glob, json, os

RES = os.path.join(os.path.dirname(__file__), "..", "results")

# --- A100-40 SXM, bf16 dense ---
C, BETA, BW = 312e12, 1.555e12, 2

# --- Llama-3.1-8B ---
N = 8.03e9
N_LAYERS, N_KV, D_H, B_KV_BYTES = 32, 8, 128, 2
KAPPA = 2 * N_LAYERS * N_KV * D_H * B_KV_BYTES      # bytes of KV per token

# --- the EAGLE3 head that actually ran (TanBaby/...-YARN-64K, model.safetensors) ---
DRAFT_BYTES = 849_765_096
N_D = DRAFT_BYTES / BW
KAPPA_D = KAPPA / N_LAYERS                           # one decoder layer of KV

t_w,  t_c  = N * BW / BETA,   2 * N / C
t_wd, t_cd = N_D * BW / BETA, 2 * N_D / C
w, w_d     = KAPPA / BETA,    KAPPA_D / BETA


def predict(B, L_in, L_out, Om, h, V, D, with_draft):
    """S_e2e from the derivation.  with_draft adds the D sequential draft passes."""
    # prefill (batch-level)
    pre_ar = max(B * t_c * L_in * (1 - h), t_w) + B * w * L_in * (1 - h)
    pre_sp = pre_ar + (max(B * t_cd * L_in * (1 - h), t_wd)
                       + B * w_d * L_in * (1 - h))
    # decode pass, at the time-averaged resident context
    L = L_in + L_out / 2
    T_ar = max(B * t_c, t_w) + B * w * (L + 1)
    T_sp = max((V + 1) * B * t_c, t_w) + B * w * (L + V + 1)
    if with_draft:
        T_sp += D * (max(B * t_cd, t_wd) + B * w_d * L)
    dec_ar = L_out * T_ar
    dec_sp = (L_out / Om) * T_sp
    return (pre_ar + dec_ar) / (pre_sp + dec_sp), pre_ar / (pre_ar + dec_ar)


def cells():
    for d in sorted(glob.glob(f"{RES}/govreport_llama_*")):
        for run in sorted(glob.glob(f"{d}/*")):
            names = {os.path.basename(p) for p in glob.glob(f"{run}/*.json")}
            if "config.json" not in names:
                continue
            cfg = json.load(open(f"{run}/config.json"))
            try:
                V = json.loads(cfg["configs"]["eagle3"])["num_speculative_tokens"]
            except Exception:
                continue
            for e in sorted(glob.glob(f"{run}/eagle3_ms*.json")):
                b = os.path.basename(e).replace("eagle3_", "baseline_")
                bp = f"{run}/{b}"
                if not os.path.exists(bp):
                    continue
                je, jb = json.load(open(e)), json.load(open(bp))
                if je.get("output_len") == 1 or jb.get("output_len") == 1:
                    continue          # prefill-only ablation, not a speedup cell
                yield os.path.basename(d), je, jb, V
