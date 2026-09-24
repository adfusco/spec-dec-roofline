"""Roofline model of speculative vs. autoregressive decode.

Implements derivation_notes_adjusted/derivation_notes.tex: each pass costs
max(compute, weight load) plus KV traffic, for the AR pass, the verify pass over
V+1 positions, and D draft passes; prefill is charged to both arms.

S() is the decode-only speedup, S_e2e() adds prefill, B_be is the batch where
speculation breaks even, and L_crit the context past which it never does.

Known defect: the KV rebate is over-credited, so B_be and L_crit are optimistic.
Measured decode speedup reaches 0.67x where this model floors at 0.91x.

Run directly to draw the phase diagram; HW=<key> picks the accelerator.
"""
import os

import numpy as np

GiB = 1073741824
# M_res: activations + CUDA context + fragmentation + attention workspace.
# 6 GiB of 48 ~= util 0.875 (vLLM defaults to 0.9); 10 GiB on the 80 GB parts.
# B_KV is +-35% over M_res in [2,10] GiB at the lowest viable TP, less higher.
# C is DENSE bf16 tensor-core throughput, not the 2:4-sparsity figure.
HW = {
    "L40S":    dict(name="L40S",        C=181e12, beta=0.864e12, bw=2, HBM_GIB=48, RES_GIB=6),
    "A100-40": dict(name="A100-40 SXM", C=312e12, beta=1.555e12, bw=2, HBM_GIB=40, RES_GIB=5),
    "A100-80": dict(name="A100-80 SXM", C=312e12, beta=2.039e12, bw=2, HBM_GIB=80, RES_GIB=10),
    "H100-80": dict(name="H100-80 SXM", C=989e12, beta=3.35e12,  bw=2, HBM_GIB=80, RES_GIB=10),
    "H200":    dict(name="H200 SXM",    C=989e12, beta=4.8e12,   bw=2, HBM_GIB=141, RES_GIB=10),
}

C_D = 0.05           # drafter as a fraction of target params (EAGLE-3 heads are ~5%)
T_OUT = 9            # output tokens assumed when sizing the KV ceiling


def ridge_batch(hw):
    """T*: batch at which a pass stops being weight-bound.  Model-independent."""
    return hw["C"] * hw["bw"] / (2 * hw["beta"])


class Model:
    """A target model's decode-pass time primitives on a given accelerator.

    Pass draft_bytes to override the default N_d = C_D*N drafter with a
    measured head size (in bytes of weights).
    """

    def __init__(s, name, N, n_layers, n_q, n_kv, d_h, hw, b_kv=2, draft_bytes=None,
                 draft_bw=None, draft_C=None, has_draft=True):
        s.name, s.N, s.n_layers, s.n_q, s.n_kv = name, N, n_layers, n_q, n_kv
        s.hw = hw
        C, beta, bw = hw["C"], hw["beta"], hw["bw"]
        s.kap   = 2 * n_layers * n_kv * d_h * b_kv
        dbw, dC = (draft_bw or bw), (draft_C or C)
        s.kap_d = s.kap / n_layers if has_draft else 0.0   # one decoder layer of KV
        s.N_d   = (((draft_bytes / dbw) if draft_bytes else C_D * N)
                   if has_draft else 0.0)
        s.draft_mem = s.N_d * dbw                          # bytes on the device
        s.t_w, s.t_c = N * bw / beta, 2 * N / C
        s.w          = s.kap / beta
        s.t_wd, s.t_cd = s.N_d * dbw / beta, 2 * s.N_d / dC
        s.w_d        = s.kap_d / beta
        s.a          = 4 * n_layers * n_q * d_h / C      # t_attn(L) = a*L
        s.a_d        = s.a / n_layers                    # 1-layer EAGLE-3 head
        s.r          = s.t_c / s.w
        s.attn  = "MHA" if n_kv == n_q else rf"GQA ($n_{{kv}}{{=}}{n_kv}$)"
        s.min_tp = next(tp for tp in (1, 2, 4, 8, 16)
                        if tp * (hw["HBM_GIB"] - hw["RES_GIB"]) * GiB > N * bw * 1.15)


def build_models(hw):
    return [Model("Phi-3-mini 3.8B",  3.8e9, 32, 32, 32,  96, hw),
            Model("Llama-2-7B",       6.7e9, 32, 32, 32, 128, hw),
            Model("Llama-2-13B",     13.0e9, 40, 40, 40, 128, hw),
            Model("Gemma-2-9B",       9.2e9, 42, 16,  8, 256, hw),
            Model("Mistral-7B",       7.2e9, 32, 32,  8, 128, hw),
            Model("Llama-3.1-8B",     8.0e9, 32, 32,  8, 128, hw),
            Model("Qwen3-14B",       14.8e9, 40, 40,  8, 128, hw),
            Model("Qwen3-32B",       32.8e9, 64, 64,  8, 128, hw),
            Model("Llama-3.1-70B",   70.6e9, 80, 64,  8, 128, hw)]


# ---------------- pass costs (derivation_notes_adjusted) ----------------
def T_ar(B, L, m):
    """One autoregressive decode pass."""
    return np.maximum(B * (m.t_c + m.a * L), m.t_w) + B * m.w * (L + 1)


def T_verify(B, L, V, m):
    """The verify pass: V+1 positions through the target's weights."""
    return np.maximum((V + 1) * B * (m.t_c + m.a * L), m.t_w) + B * m.w * (L + V + 1)


def T_draft(B, L, D, m):
    """D sequential draft passes."""
    return D * (np.maximum(B * (m.t_cd + m.a_d * L), m.t_wd) + B * m.w_d * L)


def T_spec(B, L, D, V, m):
    return T_verify(B, L, V, m) + T_draft(B, L, D, m)


def T_pre_ar(B, L_in, m, h=0.0):
    """Prefill, AR arm.  Attention is quadratic: L(L+1)/2 causal pairs."""
    comp = m.t_c * L_in + 0.5 * m.a * L_in * L_in
    return np.maximum(B * comp * (1 - h), m.t_w) + B * m.w * L_in * (1 - h)


def T_pre_spec(B, L_in, m, h=0.0):
    """Prefill, speculative arm.  The drafter cannot skip the prompt."""
    comp_d = m.t_cd * L_in + 0.5 * m.a_d * L_in * L_in
    return T_pre_ar(B, L_in, m, h) + (np.maximum(B * comp_d * (1 - h), m.t_wd)
                                      + B * m.w_d * L_in * (1 - h))


# ---------------- speedups ----------------
def S(B, L, D, V, Om, m):
    """DECODE-ONLY speedup at batch B, context L.  T_ar / (T_spec / Omega)."""
    return T_ar(B, L, m) / (T_spec(B, L, D, V, m) / Om)


def S_e2e(B, L_in, L_out, D, V, Om, m, h=0.0):
    """End-to-end speedup, prefill plus decode.  Returns (S, prefill share)."""
    L = L_in + L_out / 2.0
    pre_ar, pre_sp = T_pre_ar(B, L_in, m, h), T_pre_spec(B, L_in, m, h)
    dec_ar = L_out * T_ar(B, L, m)
    dec_sp = (L_out / Om) * T_spec(B, L, D, V, m)
    return (pre_ar + dec_ar) / (pre_sp + dec_sp), pre_ar / (pre_ar + dec_ar)


def B_be(L, D, V, Om, m, lo=1e-3, hi=1e9):
    """Largest batch at which speculation is still faster.  inf => always faster.

    OPTIMISTIC: see the KNOWN DEFECT note at the top of this file.
    """
    f = lambda B: S(B, L, D, V, Om, m) - 1.0
    if f(lo) <= 0: return 0.0
    if f(hi) >  0: return np.inf
    for _ in range(160):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return np.sqrt(lo * hi)


def S_inf(L, D, V, Om, m):
    """Decode speedup as B -> inf: the model's floor at this context.

    Measured to be breached -- 0.909x predicted vs 0.67x observed at
    GovReport 512/128, B=224.
    """
    return S(1e12, L, D, V, Om, m)


def L_crit(D, V, Om, m, lo=1.0, hi=1e7):
    """Context at which speculation breaks even even at a saturated batch.

    Solved numerically: the closed form assumed no attention term, and t_attn
    enters the verify pass (V+1) times, so it no longer factors out.
    """
    f = lambda L: S_inf(L, D, V, Om, m) - 1.0
    if f(lo) > 0: return lo
    if f(hi) <= 0: return np.inf
    for _ in range(160):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return np.sqrt(lo * hi)


def B_kv(L, m, tp):
    """Largest batch the HBM left over after weights can hold KV for."""
    hw = m.hw
    free = tp * (hw["HBM_GIB"] - hw["RES_GIB"]) * GiB - m.N * hw["bw"] - m.draft_mem
    if free <= 0: return np.nan
    return free / ((m.kap + m.kap_d) * max(1.0, tp / m.n_kv) * (L + T_OUT))


def chain(g, a):
    """Depth-g chain drafter at per-token acceptance a: (D, V, Omega)."""
    return (g, g, sum(a ** k for k in range(g + 1)))


# ============================ figure + table ============================
def main():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "font.size": 9.0, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "axes.linewidth": 0.7, "lines.linewidth": 1.8,
        "figure.dpi": 200, "savefig.bbox": "tight",
    })
    GREEN, RED, GRAY, ORNG = "#227755", "#CC4444", "#888888", "#DD8811"

    hw_key = os.environ.get("HW", "L40S")
    if hw_key not in HW:
        raise SystemExit(f"unknown HW={hw_key!r}; choose one of {sorted(HW)}")
    hw = HW[hw_key]
    gamma = int(os.environ.get("GAMMA", "4"))

    out = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(out, exist_ok=True)
    stem = f"{out}/phase_{hw_key}_g{gamma}"

    Ts = ridge_batch(hw)
    models = build_models(hw)
    by = {m.name: m for m in models}
    panels = [by["Llama-2-7B"], by["Llama-3.1-8B"], by["Qwen3-32B"], by["Llama-3.1-70B"]]

    L_LO, L_HI, YMAX, L_MAX = 32, 512, 1e5, 8192
    L = np.logspace(np.log10(16), np.log10(L_MAX), 500)
    ALPHAS = [(0.90, GREEN), (0.80, ORNG), (0.70, RED)]
    DASH = [(0, (6, 2)), (0, (3, 2)), (0, (1, 1.6))]
    clip = lambda v: np.where(np.isfinite(v), v, YMAX)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, panels):
        base = np.array([B_be(x, *chain(gamma, 0.80), m) for x in L])
        ax.fill_between(L, 1, clip(base), color=ORNG, alpha=0.13, lw=0, zorder=0)
        for al, col in ALPHAS:
            D_, V_, Om_ = chain(gamma, al)
            b = clip(np.array([B_be(x, D_, V_, Om_, m) for x in L]))
            ax.plot(L, b, color=col, label=rf"$\alpha={al:.2f}$")
            Lc = L_crit(D_, V_, Om_, m)
            if np.isfinite(Lc) and Lc < L_MAX:
                ax.plot([Lc, Lc], [b[max(np.searchsorted(L, Lc) - 1, 0)], YMAX], color=col)
        for k, (tp, dash) in enumerate(zip([m.min_tp, m.min_tp * 2, m.min_tp * 4], DASH)):
            ax.plot(L, [B_kv(x, m, tp) for x in L], color="k", ls=dash, lw=1.3,
                    label=(rf"$B_{{KV}}$, TP$\times${2**k}" if m is panels[0] else None))
        ax.axhline(Ts, color=GRAY, lw=1.0, ls=":")
        ax.axvspan(L_LO, L_HI, color=GRAY, alpha=0.15, lw=0, zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(16, L_MAX); ax.set_ylim(1, YMAX)
        ax.set_title(f"{m.name}\n{m.attn},  " rf"$r={m.r:.0f}$,  min TP$={m.min_tp}$", pad=6)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    for ax in axes[-1, :]:
        ax.set_xlabel("context length $L$ (tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("batch size $B$")
    axes[0, 0].text(np.sqrt(L_LO * L_HI), 1.7, "in-DB", ha="center", fontsize=8, color="#555555")
    axes[0, 0].text(6.5e3, Ts * 1.35, "$T^*$", color=GRAY, fontsize=9, ha="right")
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=6, columnspacing=2.0, handlelength=2.6, fontsize=9)
    fig.suptitle(r"speculation is faster inside the shaded region "
                 rf"(chain drafter, $\gamma={gamma}$; {hw['name']}, $T^*={Ts:.0f}$ everywhere)",
                 y=0.995, fontsize=10, va="top")
    fig.subplots_adjust(wspace=0.07, hspace=0.30, top=0.90, bottom=0.09)
    fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png"); plt.close(fig)

    # ---------------- table: r and L_crit across all nine models ----------------
    print(f"{hw['name']}  T*={Ts:.0f} (model-independent)  "
          f"M_res={hw['RES_GIB']} GiB of {hw['HBM_GIB']}  gamma={gamma}")
    print(f"\n{'model':16s}{'attn':>10s}{'kappa KB':>10s}{'r':>7s}{'TP':>4s}"
          f"{'Lcrit .9':>10s}{'.8':>7s}{'.7':>7s}{'B_KV(90)':>10s}{'B_be(90)':>10s}")
    for m in models:
        lc = [L_crit(*chain(gamma, al), m) for al in (0.9, 0.8, 0.7)]
        bb = B_be(90, *chain(gamma, 0.80), m)
        at = "MHA" if m.n_kv == m.n_q else f"GQA-{m.n_kv}"
        print(f"{m.name:16s}{at:>10s}{m.kap/1024:10.0f}{m.r:7.0f}{m.min_tp:4d}"
              f"{lc[0]:10.0f}{lc[1]:7.0f}{lc[2]:7.0f}"
              f"{B_kv(90, m, m.min_tp):10.0f}{bb:10.1f}")
    print(f"\nwrote {stem}.png / .pdf")


if __name__ == "__main__":
    main()
