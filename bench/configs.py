"""Sweep presets for the Modal offline-throughput benchmark runner.

Everything here is plain data, so it can be overridden from the CLI, e.g.:

    modal run bench/vllm_throughput.py --preset relqueries --max-num-seqs 32,64
"""
import json
from dataclasses import dataclass, replace


# Speculative-config strings, one per (target model, eagle3 head) pair. The head
# must match the base model exactly (tokenizer + hidden size), so each target gets
# its own. num_speculative_tokens=3 -> a linear chain of 3 draft tokens per step.
QWEN3_EAGLE3_SPEC = (
    '{"method": "eagle3", "model": "RedHatAI/Qwen3-32B-speculator.eagle3", '
    '"num_speculative_tokens": 3}'
)
QWEN3_14B_EAGLE3_SPEC = (
    '{"method": "eagle3", "model": "RedHatAI/Qwen3-14B-speculator.eagle3", '
    '"num_speculative_tokens": 3}'
)
LLAMA3_EAGLE3_SPEC = (
    '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", '
    '"num_speculative_tokens": 3}'
)
LLAMA3_EAGLE3_YARN_SPEC = (
    '{"method": "eagle3", '
    '"model": "TanBaby/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K", '
    '"num_speculative_tokens": 3}'
)

# Ungated mirror of meta-llama/Llama-3.1-8B-Instruct (identical weights/tokenizer)
# so no HF token / Modal secret is needed. Swap to the gated official repo only if
# you wire up an HF_TOKEN secret and want canonical provenance.
LLAMA3_8B = "NousResearch/Meta-Llama-3.1-8B-Instruct"


@dataclass
class ThroughputConfig:
    """One offline throughput sweep: N model configs x M batch sizes, driven through
    LLM.generate/LLM.chat (see tp_worker.py) rather than `vllm bench throughput`."""

    name: str                       # run namespace under the results volume
    target: str                     # HF model id loaded by the offline engine
    # "custom"  -> {"prompt": ...} JSONL, rendered through Qwen3's chat template
    # "sharegpt" -> ShareGPT json, first human turn sent raw (no chat template)
    dataset_kind: str
    dataset_file: str               # filename inside the datasets volume
    num_prompts: int
    max_num_seqs: list[int]         # batch-size sweep (offline analogue of concurrencies)
    # server config name -> speculative-config JSON ("" means baseline / no spec)
    configs: dict[str, str]
    gpu: str = "A100-40GB:4"        # Modal GPU request string
    tp_size: int = 4                # tensor-parallel size (match GPU count)
    n_params_b: float = 32.8        # target params (billions), for the MFU denominator
    max_model_len: int = 4096
    gpu_mem_util: float = 0.90
    # 0.0 = greedy/deterministic. At temperature 0 the output length (natural mode)
    # and spec-decode acceptance are deterministic, so runs are reproducible.
    temperature: float = 0.0
    # Only forwarded for models whose chat template supports it (e.g. Qwen3, a
    # hybrid-reasoning model); a no-op for Llama etc. Off keeps short-answer
    # relqueries out of "thinking" mode. Detected from the template in tp_worker.
    enable_thinking: bool = False
    output_mode: str = "natural"
    output_len: int = 512
    # Prefix caching on by default: relqueries share templated preambles, and a
    # real offline worker would run with APC enabled, so this reflects production.
    enable_prefix_caching: bool = True
    # Store the raw per-request prompt/output length arrays in each result JSON
    # (compact prompt_len_stats/output_len_stats are always written; this only
    # adds the full arrays, needed for summarize_lengths.py --plot histograms).
    save_lengths: bool = False
    # Store per-request timings (output.metrics) so summarize_throughput.py can
    # reconstruct true concurrency and drain-trimmed steady-state tok/s.
    save_detailed: bool = False
    # Time each engine step, and the EAGLE draft phase nested inside it, with
    # CUDA events (bench/draft_timer.py). Runs on both arms: the baseline gives
    # T_AR directly, the spec arm gives draft and (step - draft).
    draft_timer: bool = False
    # vLLM `quantization` for the TARGET ("" = unquantized). "fp8" on H100/H200
    # is W8A8 dynamic FP8, i.e. real FP8 compute. The EAGLE draft head is not
    # passed this and loads per its own config (bf16).
    quantization: str = ""
    timeout_hours: float = 4.0

    def override(self, **kw) -> "ThroughputConfig":
        """Return a copy with non-empty overrides applied (used by the CLI)."""
        clean = {k: v for k, v in kw.items() if v not in (None, "", [])}
        return replace(self, **clean)

    def with_spec_depth(self, depth: int) -> "ThroughputConfig":
        """Copy with every speculative config's draft depth set to `depth`; the run name
        gains a _d<depth> suffix."""
        configs = {
            name: (json.dumps({**json.loads(spec), "num_speculative_tokens": depth})
                   if spec else spec)
            for name, spec in self.configs.items()
        }
        return replace(self, configs=configs, name=f"{self.name}_d{depth}")


RELQUERIES_TP = ThroughputConfig(
    name="qwen_test_relqueries_tp",
    target="Qwen/Qwen3-32B",
    dataset_kind="custom",
    dataset_file="relqueries_sample_1000.jsonl",
    num_prompts=2560,
    max_num_seqs=[32, 64, 128, 256],
    configs={"baseline": "", "eagle3": QWEN3_EAGLE3_SPEC},
)

# Qwen3-32B on ShareGPT. ShareGPT is large, so N is bumped to 4000 to keep the
# steady-state window clean up to batch 512.
SHAREGPT_TP = ThroughputConfig(
    name="qwen_test_sharegpt_tp",
    target="Qwen/Qwen3-32B",
    dataset_kind="sharegpt",
    dataset_file="ShareGPT_V3_unfiltered_cleaned_split.json",
    num_prompts=4000,
    max_num_seqs=[64, 128, 256, 512],
    configs={"baseline": "", "eagle3": QWEN3_EAGLE3_SPEC},
)

# Llama-3.1-8B on relqueries. 8B fits on a single A100-40GB (weights ~16 GB) with
# plenty of KV headroom, which is the realistic in-DB deployment, so this defaults
# to TP=1 on one GPU rather than the 32B's 4-GPU layout.
LLAMA_RELQUERIES_TP = ThroughputConfig(
    name="llama8b_relqueries_tp",
    target=LLAMA3_8B,
    dataset_kind="custom",
    dataset_file="relqueries_sample_1000.jsonl",
    num_prompts=2560,
    max_num_seqs=[32, 64, 128, 256],
    configs={"baseline": "", "eagle3": LLAMA3_EAGLE3_SPEC},
    gpu="A100-40GB:1",
    tp_size=1,
    n_params_b=8.03,
)

# Llama-3.1-8B on ShareGPT (single GPU, same as above).
LLAMA_SHAREGPT_TP = ThroughputConfig(
    name="llama8b_sharegpt_tp",
    target=LLAMA3_8B,
    dataset_kind="sharegpt",
    dataset_file="ShareGPT_V3_unfiltered_cleaned_split.json",
    num_prompts=4000,
    max_num_seqs=[64, 128, 256, 512],
    configs={"baseline": "", "eagle3": LLAMA3_EAGLE3_SPEC},
    gpu="A100-40GB:1",
    tp_size=1,
    n_params_b=8.03,
)

PDMX_TP = ThroughputConfig(
    name="qwen_test_pdmx_tp",
    target="Qwen/Qwen3-32B",
    dataset_kind="custom",
    dataset_file="pdmx_sample_10000.jsonl",
    num_prompts=2560,
    max_num_seqs=[32, 64, 128, 256],
    configs={"baseline": "", "eagle3": QWEN3_EAGLE3_SPEC},
    output_len=128,
)

# Llama-3.1-8B on PDMX (single GPU).
LLAMA_PDMX_TP = ThroughputConfig(
    name="llama8b_pdmx_tp",
    target=LLAMA3_8B,
    dataset_kind="custom",
    dataset_file="pdmx_sample_10000.jsonl",
    num_prompts=2560,
    max_num_seqs=[32, 64, 128, 256],
    configs={"baseline": "", "eagle3": LLAMA3_EAGLE3_SPEC},
    gpu="A100-40GB:1",
    tp_size=1,
    n_params_b=8.03,
    output_len=128,
)


# Tiny smoke test (Qwen3-32B, eagle3 only, TP=2, few prompts/batch).
TP_DEBUG = ThroughputConfig(
    name="qwen_test_tp_debug",
    target="Qwen/Qwen3-32B",
    dataset_kind="custom",
    dataset_file="relqueries_sample_1000.jsonl",
    num_prompts=16,
    max_num_seqs=[16],
    configs={"eagle3": QWEN3_EAGLE3_SPEC},
    gpu="A100-40GB:2",
    tp_size=2,
    output_len=64,
    timeout_hours=0.5,
)

# Tiny smoke test for the Llama path (single GPU, eagle3 only).
LLAMA_DEBUG = ThroughputConfig(
    name="llama8b_tp_debug",
    target=LLAMA3_8B,
    dataset_kind="custom",
    dataset_file="relqueries_sample_1000.jsonl",
    num_prompts=16,
    max_num_seqs=[16],
    configs={"eagle3": LLAMA3_EAGLE3_SPEC},
    gpu="A100-40GB:1",
    tp_size=1,
    n_params_b=8.03,
    output_len=64,
    timeout_hours=0.5,
)


GOVREPORT_TIERS = {
    512:  ("512", 2560, [16, 32, 64, 128, 256], 309),
    1024: ("1k",  2560, [16, 32, 64, 128],      154),
    2048: ("2k",  2048, [8, 16, 32, 64],         77),
    4096: ("4k",  1024, [8, 16, 32],             39),
    8192: ("8k",   512, [4, 8, 16],              19),
}

# output target -> max_tokens cap.  Natural mode with an instruction-steered
# length; the cap is ~2x the target so it never truncates the natural stopping
# point (which would silently turn this into a fixed-length run).
GOVREPORT_OUT_CAPS = {32: 128, 128: 384, 256: 768, 512: 1024}


def _govreport_presets() -> dict[str, ThroughputConfig]:
    """One ThroughputConfig per (tokenizer, tier, output target)."""
    families = {
        "llama": dict(target=LLAMA3_8B, spec=LLAMA3_EAGLE3_YARN_SPEC,
                      gpu="A100-40GB:1", tp_size=1, n_params_b=8.03),
        "qwen":  dict(target="Qwen/Qwen3-32B", spec=QWEN3_EAGLE3_SPEC,
                      gpu="A100-40GB:4", tp_size=4, n_params_b=32.8),
        # Same Qwen3 tokenizer as Qwen3-32B, so it reads the "qwen" ladder files.
        "qwen14b": dict(target="Qwen/Qwen3-14B", spec=QWEN3_14B_EAGLE3_SPEC,
                        gpu="A100-40GB:1", tp_size=1, n_params_b=14.8, data="qwen"),
    }
    out: dict[str, ThroughputConfig] = {}
    for tag, fam in families.items():
        for budget, (label, n_prompts, seqs, _b_kv) in GOVREPORT_TIERS.items():
            for target, cap in GOVREPORT_OUT_CAPS.items():
                name = f"gr_{tag}_{label}_out{target}"
                out[name] = ThroughputConfig(
                    name=f"govreport_{tag}_{label}_out{target}",
                    target=fam["target"],
                    dataset_kind="custom",
                    dataset_file=f"govreport_{fam.get('data', tag)}_in{label}_out{target}.jsonl",
                    num_prompts=n_prompts,
                    max_num_seqs=seqs,
                    configs={"baseline": "", "eagle3": fam["spec"]},
                    gpu=fam["gpu"],
                    tp_size=fam["tp_size"],
                    n_params_b=fam["n_params_b"],
                    # room for the tier, the output cap, and template slack
                    max_model_len=budget + cap + 256,
                    output_mode="natural",
                    output_len=cap,
                )
    return out


LLAMA_PDMX_YARN_CAL = ThroughputConfig(
    name="llama8b_pdmx_yarn_cal",
    target=LLAMA3_8B,
    dataset_kind="custom",
    dataset_file="pdmx_sample_10000.jsonl",
    num_prompts=2560,
    max_num_seqs=[32],
    configs={"eagle3_yarn": LLAMA3_EAGLE3_YARN_SPEC},
    gpu="A100-40GB:1",
    tp_size=1,
    n_params_b=8.03,
    output_len=128,
    timeout_hours=1.0,
)


THROUGHPUT_PRESETS: dict[str, ThroughputConfig] = {
    "relqueries": RELQUERIES_TP,
    "sharegpt": SHAREGPT_TP,
    "pdmx": PDMX_TP,
    "llama_relqueries": LLAMA_RELQUERIES_TP,
    "llama_sharegpt": LLAMA_SHAREGPT_TP,
    "llama_pdmx": LLAMA_PDMX_TP,
    "llama_pdmx_yarn_cal": LLAMA_PDMX_YARN_CAL,
    "tp_debug": TP_DEBUG,
    "llama_debug": LLAMA_DEBUG,
}

THROUGHPUT_PRESETS.update(_govreport_presets())
