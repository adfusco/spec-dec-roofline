"""Offline, max-batch throughput sweep on Modal.

Hands the offline engine the full prompt list and sweeps max_num_seqs to find where
throughput plateaus, running a baseline and a speculative arm and summarizing each
point. No server, no TTFT/ITL, only tok/s.
"""
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import modal

from common import (
    DATASETS_DIR,
    KNOWN_SUMMARY_GPUS,
    RESULTS,
    RESULTS_DIR,
    VOLUMES,
    gpu_key,
    gpu_sampler,
    image,
)
from configs import THROUGHPUT_PRESETS, ThroughputConfig

app = modal.App("sd-db-vllm-throughput", image=image)


def _run_point(cfg: ThroughputConfig, name: str, spec: str, B: int,
               outdir: Path, dataset_path: Path, failures: Path) -> None:
    """Run one (model config, max_num_seqs) throughput point in a fresh process."""
    print(f"--- throughput {name} max_num_seqs={B} ---", flush=True)
    out_json = outdir / f"{name}_ms{B}.json"
    cmd = [
        "python", "/root/tp_worker.py",
        "--target", cfg.target,
        "--tp-size", str(cfg.tp_size),
        "--max-model-len", str(cfg.max_model_len),
        "--gpu-mem-util", str(cfg.gpu_mem_util),
        "--max-num-seqs", str(B),
        "--dataset-kind", cfg.dataset_kind,
        "--dataset-path", str(dataset_path),
        "--num-prompts", str(cfg.num_prompts),
        "--temperature", str(cfg.temperature),
        "--output-len", str(cfg.output_len),
        "--output-mode", cfg.output_mode,
        "--prefix-caching", "on" if cfg.enable_prefix_caching else "off",
        "--config", name,
        "--out-json", str(out_json),
    ]
    if spec:
        cmd += ["--spec", spec]
    if cfg.enable_thinking:
        cmd += ["--enable-thinking"]
    if cfg.save_lengths:
        cmd += ["--save-lengths"]
    if cfg.save_detailed:
        cmd += ["--save-detailed"]
    if cfg.quantization:
        cmd += ["--quantization", cfg.quantization]
    if cfg.draft_timer:
        # Also on the baseline arm: execute_model is timed with or without
        # speculation, and that is where T_AR comes from as a measurement
        # rather than as (elapsed - prefill ablation) / assumed step count.
        cmd += ["--draft-timer"]

    log = outdir / f"tp_{name}_ms{B}.log"
    with gpu_sampler(outdir / f"gpu_samples_{name}_ms{B}.csv"):
        with open(log, "wb") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        msg = f"!! throughput failed for {name} max_num_seqs={B}, see {log.name}\n"
        print(msg, flush=True)
        failures.open("a").write(msg)
    RESULTS.commit()


def _one_sweep(cfg: ThroughputConfig, outdir: Path, dataset_path: Path) -> None:
    """Run every model config x max_num_seqs once into `outdir`, then summarize."""
    outdir.mkdir(parents=True, exist_ok=True)
    failures = outdir / "failures.log"
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    for name, spec in cfg.configs.items():
        for B in cfg.max_num_seqs:
            _run_point(cfg, name, spec, B, outdir, dataset_path, failures)

    summary = outdir / "summary.txt"
    summarize_args = ["python", "/root/summarize_throughput.py", str(outdir),
                      "--tp", str(cfg.tp_size), "--n-params-b", str(cfg.n_params_b)]
    key = gpu_key(cfg.gpu)
    if key in KNOWN_SUMMARY_GPUS:
        summarize_args += ["--gpu", key]  # else summarize_throughput.py defaults to a100-40gb
    with open(summary, "wb") as f:
        subprocess.run(summarize_args, stdout=f, stderr=subprocess.STDOUT, check=False)
    print(summary.read_text() if summary.exists() else "(no summary)", flush=True)

    # Workload length profile (input/output token distributions), kept separate
    # from the throughput table above.
    len_summary = outdir / "summary_lengths.txt"
    with open(len_summary, "wb") as f:
        subprocess.run(["python", "/root/summarize_lengths.py", str(outdir)],
                       stdout=f, stderr=subprocess.STDOUT, check=False)
    print(len_summary.read_text() if len_summary.exists() else "(no length summary)", flush=True)
    RESULTS.commit()


def _remote_sweep(cfg_dict: dict, repeats: int = 1, batch_id: str = "") -> str:
    """Container entrypoint: run the throughput sweep `repeats` times.

    Layout convention: repeats==1 keeps the flat `<name>/<run_id>/`; repeats>1
    groups runs under `<name>/<batch_id>/rep<i>/`.
    """
    cfg = ThroughputConfig(**cfg_dict)

    dataset_path = Path(DATASETS_DIR) / cfg.dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset not found at {dataset_path}; run "
            f"`modal run bench/prepare_datasets.py` first"
        )

    batch_id = batch_id or time.strftime("%Y%m%d_%H%M%S")
    base = Path(RESULTS_DIR) / cfg.name / batch_id
    for i in range(1, repeats + 1):
        outdir = base if repeats == 1 else base / f"rep{i}"
        print(f"===== repeat {i}/{repeats} -> {outdir} =====", flush=True)
        _one_sweep(cfg, outdir, dataset_path)

    print(f"done -> {base}", flush=True)
    return str(base)


# Exposed as a Cls (not a plain function) because in Modal >=1.x only `Cls`
# supports `.with_options(...)`, which the entrypoint uses to re-spec the GPU and
# timeout per preset/CLI at call time.
@app.cls(volumes=VOLUMES, gpu="A100-40GB:4", timeout=4 * 3600)
class Throughput:
    @modal.method()
    def run(self, cfg_dict: dict, repeats: int = 1, batch_id: str = "") -> str:
        return _remote_sweep(cfg_dict, repeats, batch_id)


@app.local_entrypoint()
def main(
    preset: str = "relqueries",
    repeats: int = 1,
    gpu: str = "",
    tp_size: int = 0,
    num_prompts: int = 0,
    max_num_seqs: str = "",
    max_model_len: int = 0,
    gpu_mem_util: float = 0.0,
    temperature: float = -1.0,
    output_len: int = 0,
    output_mode: str = "",
    spec_depth: int = 0,
    no_prefix_caching: bool = False,
    thinking: bool = False,
    save_lengths: bool = False,
    save_detailed: bool = False,
    draft_timer: bool = False,
    quantization: str = "",
    dataset_file: str = "",
):
    """Launch an offline throughput sweep on Modal.

    --preset picks the base config; other flags override it (--max-num-seqs
    128,256 sweeps batch, --spec-depth sets the draft chain length, --quantization
    sets the target dtype, --draft-timer records per-step CUDA-event timings)."""
    if preset not in THROUGHPUT_PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {list(THROUGHPUT_PRESETS)}")
    if repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if spec_depth < 0:
        raise SystemExit("--spec-depth must be >= 1")
    if output_mode and output_mode not in ("natural", "fixed"):
        raise SystemExit("--output-mode must be 'natural' or 'fixed'")

    seqs = [int(x) for x in max_num_seqs.split(",") if x.strip()] if max_num_seqs else None
    cfg = THROUGHPUT_PRESETS[preset].override(
        gpu=gpu,
        tp_size=tp_size or None,
        num_prompts=num_prompts or None,
        max_num_seqs=seqs,
        max_model_len=max_model_len or None,
        gpu_mem_util=gpu_mem_util or None,
        temperature=temperature if temperature >= 0 else None,  # <0 = keep preset (default 0.0)
        output_len=output_len or None,
        output_mode=output_mode or None,
        dataset_file=dataset_file or None,
    )
    if spec_depth:  # draft chain length D (default: whatever the spec strings carry)
        cfg = cfg.with_spec_depth(spec_depth)
    if thinking:  # opt back into Qwen3 thinking mode (default off)
        cfg = cfg.override(enable_thinking=True)
    if no_prefix_caching:  # measure raw compute without APC (default: APC on)
        cfg = cfg.override(enable_prefix_caching=False)
    if save_lengths:  # keep raw per-request length arrays (for --plot histograms)
        cfg = cfg.override(save_lengths=True)
    if save_detailed:  # keep per-request timings (for steady-state/concurrency)
        cfg = cfg.override(save_detailed=True)
    if draft_timer:  # CUDA-event timing of the EAGLE draft phase (spec configs only)
        cfg = cfg.override(draft_timer=True)
    if quantization:  # target quantization, e.g. fp8; run lands under <name>_<quant>
        cfg = cfg.override(quantization=quantization, name=f"{cfg.name}_{quantization}")

    print(f"launching throughput preset={preset} x{repeats} on {cfg.gpu} "
          f"(tp={cfg.tp_size}) name={cfg.name} configs={list(cfg.configs)} "
          f"spec_depth={spec_depth or 'preset default'} temp={cfg.temperature} "
          f"thinking={cfg.enable_thinking} out_mode={cfg.output_mode} "
          f"out_len={cfg.output_len} apc={cfg.enable_prefix_caching} "
          f"max_num_seqs={cfg.max_num_seqs} "
          f"draft_timer={cfg.draft_timer} quant={cfg.quantization or 'none'} "
          f"dataset={cfg.dataset_file} prompts={cfg.num_prompts}", flush=True)

    timeout_s = int(repeats * cfg.timeout_hours * 3600)
    tp = Throughput.with_options(gpu=cfg.gpu, timeout=timeout_s)

    # .spawn() decouples the multi-hour GPU run from this local client, so a
    # network/DNS blip or closing the laptop can't cancel a detached run.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    rel = Path(cfg.name) / run_id
    fc = tp().run.spawn(asdict(cfg), repeats, run_id)

    print(f"\nsubmitted FunctionCall: {fc.object_id}")
    print(f"results in volume 'sd-db-results' at: {RESULTS_DIR}/{rel}")
    print("follow logs:       modal app logs sd-db-vllm-throughput")
    # Destination must be the preset dir: `modal volume get` keeps only the
    # final path component, so `./results` would flatten to bare <run_id>/.
    print(f"pull when done:    modal volume get sd-db-results '{rel}' ./results/{cfg.name}")
    print("\nLaunch with `modal run --detach ...` so the job survives disconnects. "
          "Waiting for completion (Ctrl-C to stop watching; the run continues)...")
    try:
        fc.get()
        print(f"\ndone -> {RESULTS_DIR}/{rel}")
    except KeyboardInterrupt:
        print(f"\nstopped watching; job still running as {fc.object_id}. "
              "Reattach with:  modal app logs sd-db-vllm-throughput")
