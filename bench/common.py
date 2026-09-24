"""Shared Modal infra for the offline-throughput benchmark runner.

`vllm_throughput.py` measures offline max-batch throughput (`LLM.generate` over
the whole prompt list): aggregate tok/s vs. `max_num_seqs`, i.e. the metric that
matters for offline DB-query batch processing where per-request latency is
irrelevant.

Everything shared (image, volumes, GPU sampler, GPU-key helper) lives here.
"""
import contextlib
import subprocess
from pathlib import Path

import modal

# --- volumes (persist across runs) ------------------------------------------
HF_CACHE = modal.Volume.from_name("sd-db-hf-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("sd-db-results", create_if_missing=True)
DATASETS = modal.Volume.from_name("sd-db-datasets", create_if_missing=True)

HF_CACHE_DIR = "/cache/hf"
RESULTS_DIR = "/results"
DATASETS_DIR = "/datasets"

VOLUMES = {HF_CACHE_DIR: HF_CACHE, RESULTS_DIR: RESULTS, DATASETS_DIR: DATASETS}

# GPU keys the summarizers know how to price (peak HBM bandwidth / FLOPs).
KNOWN_SUMMARY_GPUS = {"a100-40gb", "a100-80gb", "h100", "h100-nvl", "h200"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.23.0",
        "hf-transfer",
        "pandas",
        "tabulate",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_CACHE_ROOT": f"{RESULTS_DIR}/.vllm_cache",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .run_commands(
        "mkdir -p /opt/sd_draft_plugin",
        "cat > /opt/sd_draft_plugin/pyproject.toml <<'TOML'\n"
        "[build-system]\n"
        'requires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "[project]\n"
        'name = "sd-draft-plugin"\n'
        'version = "0.1"\n'
        '[project.entry-points."vllm.general_plugins"]\n'
        'sd_draft_timer = "sd_draft_plugin:register"\n'
        "[tool.setuptools]\n"
        'py-modules = ["sd_draft_plugin"]\n'
        "TOML",
        "cat > /opt/sd_draft_plugin/sd_draft_plugin.py <<'PY'\n"
        "def register():\n"
        "    import os, sys\n"
        '    if not os.environ.get("SD_DRAFT_TIMER_PATH", "").strip():\n'
        "        return\n"
        '    if "/root" not in sys.path:\n'
        '        sys.path.insert(0, "/root")\n'
        "    try:\n"
        "        import draft_timer\n"
        "        draft_timer.install()\n"
        "    except Exception as exc:\n"
        '        print(f"[sd_draft_plugin] {exc}", flush=True)\n'
        "PY",
        "pip install --no-deps /opt/sd_draft_plugin",
    )
    .add_local_python_source("common", "configs")
    .add_local_file("analysis/summarize_throughput.py", "/root/summarize_throughput.py")
    .add_local_file("analysis/summarize_lengths.py", "/root/summarize_lengths.py")
    .add_local_file("bench/tp_worker.py", "/root/tp_worker.py")
    .add_local_file("bench/draft_timer.py", "/root/draft_timer.py")
)


def gpu_key(gpu: str) -> str:
    """Modal GPU request -> summarizer GPU key ('A100-40GB:4' -> 'a100-40gb')."""
    return gpu.split(":")[0].lower()


@contextlib.contextmanager
def gpu_sampler(dest: Path):
    """Sample GPU util/mem/power at 1 Hz into `dest` for the duration of the block."""
    with open(dest, "wb") as f:
        proc = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader", "-l", "1"],
            stdout=f, stderr=subprocess.DEVNULL,
        )
        try:
            yield
        finally:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
