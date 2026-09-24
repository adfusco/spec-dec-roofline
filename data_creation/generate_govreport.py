"""GovReport context-ladder builder.

One dataset file per (tokenizer, input tier, output target) from
ccdv/govreport-summarization. The same documents appear at every tier, so only the
truncation budget changes and L_in varies while the corpus does not.
"""
import argparse
import json
from pathlib import Path

HF_DATASET_ID = "ccdv/govreport-summarization"
DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

# Short tag used in filenames, per tokenizer family.  Keeps the Llama and Qwen
# ladders separate: the same text tokenizes to different lengths, so one file
# cannot be exact for both.
TOKENIZER_TAGS = {
    "NousResearch/Meta-Llama-3.1-8B-Instruct": "llama",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "Qwen/Qwen3-32B": "qwen",
}

# input-token tier -> (label, number of documents to emit)
# Counts fall with tier size: the sweeps need N >> batch for a clean steady
# state, and the reachable batch shrinks with context (B_KV).
TIERS = {
    512:  ("512", 2560),
    1024: ("1k",  2560),
    2048: ("2k",  2048),
    4096: ("4k",  1024),
    8192: ("8k",   512),
}

# target output tokens -> instruction fragment.  ~0.75 tokens/word, so the word
# counts aim a little under the token target.
OUT_TARGETS = {
    32:  "in a single sentence of about 25 words",
    128: "in about 100 words",
    256: "in about 200 words",
    512: "in about 400 words",
}

PREAMBLE = (
    "You are summarizing a U.S. government report for an analyst. "
    "The report text follows.\n\n"
)


def tokenizer_tag(model: str) -> str:
    """Filename tag for a tokenizer; falls back to a sanitized model name."""
    if model in TOKENIZER_TAGS:
        return TOKENIZER_TAGS[model]
    return model.split("/")[-1].lower().replace(".", "").replace("-", "")[:12]


def tier_filename(tag: str, budget: int, out_target: int) -> str:
    label, _ = TIERS[budget]
    return f"govreport_{tag}_in{label}_out{out_target}.jsonl"


def build_prompt(body: str, out_target: int) -> str:
    """Assemble the full user message for one document."""
    return f"{PREAMBLE}{body}\n\nSummarize the report above {OUT_TARGETS[out_target]}."


def n_tokens(encoded) -> int:
    """Token count from whatever apply_chat_template returned.

    transformers <5 returns a flat list of ids; >=5 returns a BatchEncoding
    (return_dict defaults to True), where len() counts *keys* -- 2 -- not tokens.
    That failure is silent and uniform, so normalize the shape explicitly.
    """
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if len(encoded) and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]          # batched: [[ids]]
    return len(encoded)


def chat_len(tokenizer, text: str) -> int:
    """Length of `text` as tp_worker actually sends it (chat template applied)."""
    return n_tokens(tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=True,
    ))


def scaffold_overhead(tokenizer, out_target: int) -> int:
    """Tokens the prompt costs with an empty report body.

    Everything except the report itself: chat template, preamble, and the length
    directive.  Subtracted from the tier budget so the *templated* prompt lands
    on the tier, not the raw body.
    """
    return chat_len(tokenizer, build_prompt("", out_target))


def truncate_to_tokens(tokenizer, text: str, n_tokens: int) -> str:
    """Truncate `text` to at most `n_tokens` of this tokenizer, returned as text.

    clean_up_tokenization_spaces is forced off: transformers warns it is
    destructive for BPE tokenizers (it strips spaces before punctuation), and
    every document in every tier passes through this decode.
    """
    if n_tokens <= 0:
        return ""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= n_tokens:
        return text
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def build_govreport_datasets(docs, tokenizer, budgets, out_targets, out_dir,
                             tag, log=print):
    """Write one file per (budget, out_target) from an iterable of report strings.

    `docs` is consumed once and buffered: the tiers share documents, so the same
    text is truncated to each budget rather than drawing a fresh sample per tier.
    Returns a summary dict keyed by filename.
    """
    budgets = sorted(budgets)
    out_targets = sorted(out_targets)
    need = max(TIERS[b][1] for b in budgets)

    max_body = max(budgets) - min(scaffold_overhead(tokenizer, t) for t in out_targets)
    reports, seen, short = [], 0, 0
    for text in docs:
        text = (text or "").strip()
        if not text:
            continue
        seen += 1
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < max_body:
            short += 1
            continue
        reports.append(tokenizer.decode(ids[:max_body], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False))
        if len(reports) >= need:
            break
    log(f"buffered {len(reports)} reports (needed {need}); "
        f"scanned {seen}, dropped {short} shorter than {max_body} tokens")
    if len(reports) < need:
        log(f"  WARNING: only {len(reports)} documents are long enough for the "
            f"{max(budgets)}-token tier; lower tiers will be short of their "
            f"configured counts")

    summary = {}
    for budget in budgets:
        label, n_docs = TIERS[budget]
        pool = reports[:n_docs]
        for out_target in out_targets:
            over = scaffold_overhead(tokenizer, out_target)
            body_budget = budget - over
            if body_budget <= 0:
                log(f"  SKIP in{label}/out{out_target}: scaffold ({over} tok) "
                    f"exceeds the {budget} tok tier")
                continue
            fname = tier_filename(tag, budget, out_target)
            lens = []
            with open(Path(out_dir) / fname, "w") as f:
                for text in pool:
                    body = truncate_to_tokens(tokenizer, text, body_budget)
                    prompt = build_prompt(body, out_target)
                    f.write(json.dumps({"prompt": prompt}) + "\n")
                    lens.append(chat_len(tokenizer, prompt))
            lens.sort()
            summary[fname] = dict(
                n=len(lens), target=budget, scaffold=over,
                mean=round(sum(lens) / max(len(lens), 1), 1),
                p50=lens[len(lens) // 2] if lens else 0,
                p99=lens[int(len(lens) * 0.99)] if lens else 0,
                max=lens[-1] if lens else 0,
            )
            s = summary[fname]
            log(f"  {fname}: n={s['n']} target={budget} "
                f"mean={s['mean']} p50={s['p50']} p99={s['p99']} max={s['max']}")
    return summary


def verify_tiers(paths, tokenizer, log=print):
    """Re-measure chat-templated lengths of already-written tier files."""
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            log(f"  {p.name}: MISSING")
            continue
        lens = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    lens.append(chat_len(tokenizer, json.loads(line)["prompt"]))
        lens.sort()
        out[p.name] = dict(
            n=len(lens),
            mean=round(sum(lens) / max(len(lens), 1), 1),
            p50=lens[len(lens) // 2] if lens else 0,
            p99=lens[int(len(lens) * 0.99)] if lens else 0,
        )
        log(f"  {p.name}: {out[p.name]}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="tokenizer to budget against (also picks the filename tag)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tiers", default="512,1024,2048,4096,8192")
    ap.add_argument("--out-targets", default="32,128,512")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(HF_DATASET_ID, split=args.split)
    budgets = [int(x) for x in args.tiers.split(",") if x.strip()]
    targets = [int(x) for x in args.out_targets.split(",") if x.strip()]

    build_govreport_datasets(
        (r["report"] for r in ds), tok, budgets, targets,
        args.out_dir, tokenizer_tag(args.model))


if __name__ == "__main__":
    main()
