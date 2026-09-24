"""One-time dataset prep for the Modal benchmark runner.

Populates the sd-db-datasets volume with ShareGPT, the relqueries sample, PDMX, and
the GovReport input-length ladder.
"""
import glob
import json
import os
import random
import sys
import tempfile
import urllib.request
from pathlib import Path

import modal

# pure helpers can be packaged into builder images via add_local_python_source.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_creation"))

DATASETS = modal.Volume.from_name("sd-db-datasets", create_if_missing=True)
DATASETS_DIR = "/datasets"

QWEN3_32B = "Qwen/Qwen3-32B"
LLAMA3_8B = "NousResearch/Meta-Llama-3.1-8B-Instruct"

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
RELQUERIES_FILE = "relqueries_sample_1000.jsonl"

image = modal.Image.debian_slim(python_version="3.12")
app = modal.App("sd-db-prepare-datasets", image=image)


@app.function(volumes={DATASETS_DIR: DATASETS}, timeout=1800)
def fetch_sharegpt(force: bool = False) -> str:
    """Download the ShareGPT dataset into the datasets volume (once)."""
    dest = Path(DATASETS_DIR) / SHAREGPT_FILE
    if dest.exists() and not force:
        print(f"ShareGPT already present ({dest.stat().st_size} bytes), skipping")
        return str(dest)

    print(f"downloading ShareGPT -> {dest} ...", flush=True)
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(SHAREGPT_URL, tmp)
    tmp.rename(dest)
    DATASETS.commit()
    print(f"done ({dest.stat().st_size} bytes)")
    return str(dest)


# GovReport context ladder: needs `datasets` (the HF repo is plain parquet) plus
# transformers for exact, tokenizer-specific truncation.
govreport_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers>=4.44", "datasets>=2.19", "huggingface_hub", "hf_xet",
                 # apply_chat_template renders a Jinja template; the slim image
                 # has no jinja2 transitively the way the vLLM image does.
                 "jinja2")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("generate_govreport")
)


@app.function(image=govreport_image, volumes={DATASETS_DIR: DATASETS}, timeout=2 * 3600)
def build_govreport_ladder(model: str, tiers: str, out_targets: str, split: str,
                           verify: bool = False) -> dict:
    """Build the GovReport input-length ladder for one tokenizer into the volume.

    One file per (tier, output target).  The tiers share documents -- the same
    reports truncated to different budgets -- so L_in varies while the corpus
    does not.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import generate_govreport as gg

    tok = AutoTokenizer.from_pretrained(model)
    tag = gg.tokenizer_tag(model)
    budgets = [int(x) for x in tiers.split(",") if x.strip()]
    targets = [int(x) for x in out_targets.split(",") if x.strip()]

    print(f"loading {gg.HF_DATASET_ID} split={split} ...", flush=True)
    ds = load_dataset(gg.HF_DATASET_ID, split=split)
    print(f"  {len(ds)} documents; tokenizer={model} (tag={tag})", flush=True)

    summary = gg.build_govreport_datasets(
        (r["report"] for r in ds), tok, budgets, targets, DATASETS_DIR, tag)
    DATASETS.commit()

    if verify:
        print("verifying written tiers (chat-templated lengths) ...", flush=True)
        gg.verify_tiers([Path(DATASETS_DIR) / n for n in summary], tok)
    print(f"govreport ladder committed: {len(summary)} files")
    return summary


def _build_relqueries_sample(n: int, seed: int, out_path: Path) -> int:
    """Flatten data_creation/*_10k_relqueries.jsonl into `n` {"prompt": ...} rows."""
    here = Path(__file__).resolve().parent
    src_dir = here.parent / "data_creation"
    files = sorted(glob.glob(str(src_dir / "*_10k_relqueries.jsonl")))
    if not files:
        raise FileNotFoundError(f"no *_10k_relqueries.jsonl found in {src_dir}")

    prompts: list[str] = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for key, val in row.items():
                    if key.startswith("prompt_") and isinstance(val, str) and val.strip():
                        prompts.append(val)

    random.Random(seed).shuffle(prompts)
    prompts = prompts[:n]
    with open(out_path, "w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    return len(prompts)


@app.local_entrypoint()
def main(n_relqueries: int = 1000, n_pdmx: int = 10000, seed: int = 42,
         force_sharegpt: bool = False,
         build_govreport: bool = False, only_govreport: bool = False,
         govreport_models: str = f"{LLAMA3_8B},{QWEN3_32B}",
         govreport_tiers: str = "512,1024,2048,4096,8192",
         govreport_out_targets: str = "32,128,512",
         govreport_split: str = "train", govreport_verify: bool = False):
    if not only_govreport:
        _prepare_core(n_relqueries, n_pdmx, seed, force_sharegpt)

    if build_govreport or only_govreport:
        # One ladder per tokenizer: the same text tokenizes to different lengths
        # under Llama and Qwen, so a shared file cannot be exact for both.
        for model in [m.strip() for m in govreport_models.split(",") if m.strip()]:
            print(f"building GovReport ladder for {model} "
                  f"(tiers={govreport_tiers}, out_targets={govreport_out_targets}) ...")
            summary = build_govreport_ladder.remote(
                model, govreport_tiers, govreport_out_targets,
                govreport_split, govreport_verify)
            print(f"uploaded {len(summary)} GovReport files for {model}")

    print("datasets ready.")


def _prepare_core(n_relqueries: int, n_pdmx: int, seed: int, force_sharegpt: bool):
    # 1) Build the relqueries sample locally, then upload it to the volume.
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / RELQUERIES_FILE
        count = _build_relqueries_sample(n_relqueries, seed, local)
        print(f"built {RELQUERIES_FILE} with {count} prompts, uploading ...")
        with DATASETS.batch_upload(force=True) as batch:
            batch.put_file(str(local), f"/{RELQUERIES_FILE}")
    print(f"uploaded {RELQUERIES_FILE} to volume 'sd-db-datasets'")

    # 2) Build the PDMX sample locally (from data_creation/PDMX.csv) and upload.
    #    Imported here (not at module top) so the remote container -- which has no
    #    data_creation/ mounted -- never tries to import it.
    import sys
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "data_creation"))
    from generate_pdmx import build_pdmx_sample

    csv_path = here.parent / "data_creation" / "PDMX.csv"
    pdmx_file = f"pdmx_sample_{n_pdmx}.jsonl"
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / pdmx_file
        count = build_pdmx_sample(str(csv_path), n_pdmx, seed, local)
        print(f"built {pdmx_file} with {count} prompts, uploading ...")
        with DATASETS.batch_upload(force=True) as batch:
            batch.put_file(str(local), f"/{pdmx_file}")
    print(f"uploaded {pdmx_file} to volume 'sd-db-datasets'")

    # 3) Fetch ShareGPT remotely straight into the volume.
    fetch_sharegpt.remote(force=force_sharegpt)
