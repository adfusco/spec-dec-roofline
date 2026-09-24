"""Build an offline-benchmark prompt dataset from PDMX.csv.

Each sampled row becomes one {"prompt": ...} line: a fixed data-analyst preamble, a
fixed music-summary query, and the row's fields serialized as JSON.
"""
import argparse
import csv
import json
import random
from pathlib import Path

# PDMX has no pathologically large cells, but raise the field cap defensively so
# an unexpected long value can't abort the read.
csv.field_size_limit(10 ** 9)

# Fixed user query (the {QUERY} slot), identical for every row.
QUERY = (
    "Given the following fields, provide an overview on the music type, and "
    "analyze the given scores. Give exactly 50 words of summary."
)

# Fixed system preamble + query + per-row data. Only {fields} varies row to row.
PROMPT_TEMPLATE = (
    "You are a data analyst. Use the provided JSON data to answer the user "
    "query based on the specified fields. Respond with only the answer, no "
    "extra formatting.\n\n"
    "Answer the below query: {QUERY}\n\n"
    "Given the following data: {fields}"
)


def _coerce(v):
    """CSV cells are strings; coerce ints/floats so the JSON reads naturally,
    leaving everything else ('NA', 'True'/'False', text) as-is. Empty -> Unknown."""
    if v is None or v == "":
        return "Unknown"
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def build_prompt(row: dict) -> str:
    """One full prompt string for a PDMX row (all fields, no trimming)."""
    fields = {k: _coerce(v) for k, v in row.items()}
    return PROMPT_TEMPLATE.format(
        QUERY=QUERY, fields=json.dumps(fields, ensure_ascii=False)
    )


def build_pdmx_sample(csv_path, n: int, seed: int, out_path) -> int:
    """Reservoir-sample up to `n` rows from `csv_path` (reproducible via `seed`)
    and write one {"prompt": ...} line each to `out_path`. Returns the count."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if len(reservoir) < n:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = row
    with open(out_path, "w") as f:
        for row in reservoir:
            f.write(json.dumps({"prompt": build_prompt(row)}) + "\n")
    return len(reservoir)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", default=str(here / "PDMX.csv"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="output JSONL (default: pdmx_sample_<n>.jsonl beside this script)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else here / f"pdmx_sample_{args.n}.jsonl"
    count = build_pdmx_sample(args.csv, args.n, args.seed, out)
    print(f"wrote {count} prompts -> {out}")


if __name__ == "__main__":
    main()
