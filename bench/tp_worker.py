"""In-container worker: run ONE offline throughput point and write its JSON.

Invoked once per (model config, max_num_seqs) by vllm_throughput.py in a fresh
subprocess, so each point gets the whole GPU with no residual engine state.
"""
import argparse
import json
import os
import statistics as st
import time
from pathlib import Path


def len_stats(lengths: list[int], cap: int | None = None) -> dict:
    """Compact length distribution: n, mean, std, min, p50/p90/p99, max."""
    if not lengths:
        return {}
    s = sorted(lengths)
    n = len(s)
    q = lambda p: s[min(n - 1, int(round(p * (n - 1))))]
    d = {
        "n": n,
        "mean": round(st.mean(s), 1),
        "std": round(st.pstdev(s), 1),
        "min": s[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": s[-1],
    }
    if cap is not None:
        d["clipped_frac"] = round(sum(1 for x in s if x >= cap) / n, 4)
    return d


def _detailed_timings(outs) -> dict:
    """Per-request arrival/first-token/finish times, for steady-state throughput."""
    try:
        m = [o.metrics for o in outs]
        if any(x is None for x in m):
            print("save-detailed: output.metrics is None on this vLLM build; skipping",
                  flush=True)
            return {}

        def pick(o, *names):
            for n in names:
                v = getattr(o, n, None)
                if v is not None:
                    return v
            return None

        # V1's RequestStateStats uses monotonic *_ts fields; older builds used
        # first_scheduled_time/first_token_time/finished_time. Accept either.
        sched = [pick(x, "scheduled_ts", "first_scheduled_time") for x in m]
        ftok = [pick(x, "first_token_ts", "first_token_time") for x in m]
        last = [pick(x, "last_token_ts", "finished_time") for x in m]
        if any(s is None for s in sched) or any(e is None for e in last):
            print("save-detailed: timing fields missing on metrics object; skipping",
                  flush=True)
            return {}
        olens = [len(o.outputs[0].token_ids) if o.outputs else 0 for o in outs]
        return {
            "first_scheduled": sched,
            "first_token": ftok,
            "finished": last,
            "output_lens": olens,
        }
    except Exception as e:
        print(f"save-detailed: could not read timings ({e}); skipping", flush=True)
        return {}


def load_prompts(kind: str, path: str, n: int) -> list[str]:
    """Return up to `n` raw prompt strings from the dataset file."""
    prompts: list[str] = []
    if kind == "custom":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line).get("prompt")
                if p and p.strip():
                    prompts.append(p)
                if len(prompts) >= n:
                    break
    elif kind == "sharegpt":
        # ShareGPT: list of conversations; take the first human turn as the
        # prompt (same content the serve sharegpt loader sends).
        with open(path) as f:
            data = json.load(f)
        for conv in data:
            turns = conv.get("conversations") or []
            if turns and turns[0].get("from") == "human" and turns[0].get("value", "").strip():
                prompts.append(turns[0]["value"])
            if len(prompts) >= n:
                break
    else:
        raise SystemExit(f"unknown dataset kind {kind!r}")
    return prompts[:n]


def prefix_cache_metrics(llm) -> dict:
    """Prefix-cache queries/hits from the engine, if exposed by this vLLM build."""
    try:
        queries = hits = 0
        for m in llm.get_metrics():
            if m.name == "vllm:prefix_cache_queries":
                queries += m.value
            elif m.name == "vllm:prefix_cache_hits":
                hits += m.value
        if queries <= 0:
            return {}
        return {
            "prefix_cache_queries": int(queries),
            "prefix_cache_hits": int(hits),
            "prefix_cache_hit_rate": hits / queries,
        }
    except Exception as e:                                 # metrics API is version-dependent
        print(f"prefix cache metrics unavailable: {e}", flush=True)
        return {}


def acceptance_metrics(llm, num_spec: int) -> dict:
    """Spec-decode acceptance: mean accepted+bonus per step, and per-position rates."""
    try:
        n_drafts = n_accepted = 0
        per_pos = [0.0] * num_spec if num_spec else []
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_drafts":
                n_drafts += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens":
                n_accepted += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                for i in range(min(len(m.values), len(per_pos))):
                    per_pos[i] += m.values[i]
        if n_drafts <= 0:
            return {}
        return {
            "acceptance_length": 1 + n_accepted / n_drafts,
            "per_pos_acceptance": [c / n_drafts for c in per_pos],
            "num_drafts": int(n_drafts),
            "num_accepted_tokens": int(n_accepted),
        }
    except Exception as e:                                 # metrics API is version-dependent
        print(f"acceptance metrics unavailable: {e}", flush=True)
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tp-size", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--gpu-mem-util", type=float, required=True)
    ap.add_argument("--max-num-seqs", type=int, required=True)
    ap.add_argument("--dataset-kind", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output-len", type=int, default=512)     # exact length (fixed) or cap (natural)
    ap.add_argument("--output-mode", choices=["natural", "fixed"], default="natural")
    ap.add_argument("--prefix-caching", choices=["on", "off"], default="on")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--spec", default="")          # speculative-config JSON, "" = baseline
    ap.add_argument("--config", required=True)      # config name (baseline/eagle3/...)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--save-lengths", action="store_true",
                    help="also store the raw per-request prompt/output length arrays "
                         "(enables histograms in summarize_lengths.py --plot)")
    ap.add_argument("--save-detailed", action="store_true",
                    help="also store per-request timings from output.metrics "
                         "(first_scheduled/first_token/finished) so the analyzer can "
                         "reconstruct true concurrency and drain-trimmed steady-state tok/s")
    ap.add_argument("--draft-timer", action="store_true",
                    help="time the EAGLE draft phase per engine step with CUDA "
                         "events (see draft_timer.py); writes <out>_draft_pid*.csv")
    ap.add_argument("--quantization", default="",
                    help="vLLM quantization for the target model, e.g. fp8")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    spec = json.loads(args.spec) if args.spec else None

    if args.draft_timer:
        os.environ["SD_DRAFT_TIMER_PATH"] = str(Path(f"{out_stem}_draft.csv"))
        try:
            import draft_timer
            draft_timer.install()
        except Exception as exc:
            print(f"draft timer unavailable: {exc}", flush=True)
    else:
        os.environ.pop("SD_DRAFT_TIMER_PATH", None)

    llm_kwargs = dict(
        model=args.target,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        speculative_config=spec,
        enable_prefix_caching=(args.prefix_caching == "on"),
        disable_log_stats=False,
        seed=0,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    prompts = load_prompts(args.dataset_kind, args.dataset_path, args.num_prompts)
    if not prompts:
        raise SystemExit(f"no prompts loaded from {args.dataset_path}")

    # Drop prompts that wouldn't leave room for the output so one over-long
    # prompt can't fail the whole batch. custom uses chat() (template overhead);
    # sharegpt sends the raw first turn through generate().
    tok = llm.get_tokenizer()
    budget = args.max_model_len - args.output_len - 16
    n_before = len(prompts)
    if args.dataset_kind == "sharegpt":
        prompts = [p for p in prompts
                   if len(tok(p, add_special_tokens=True).input_ids) <= budget]
    else:
        # Measure the chat-templated length (what llm.chat actually sends).
        def _chat_len(p: str) -> int:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=True,
            )
            # transformers <5 returns a flat id list; >=5 returns a BatchEncoding,
            # whose len() is the number of keys (2), which would silently disable
            # this whole filter.
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, dict):
                ids = ids["input_ids"]
            if len(ids) and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            return len(ids)
        prompts = [p for p in prompts if _chat_len(p) <= budget]
    if len(prompts) < n_before:
        print(f"dropped {n_before - len(prompts)}/{n_before} prompts over "
              f"context budget {budget}", flush=True)
    if not prompts:
        raise SystemExit("all prompts exceeded the context budget")

    # Natural: stop at EOS, output_len is just a cap -> real workload lengths.
    # Fixed: ignore_eos, every request emits exactly output_len tokens.
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_len,
        ignore_eos=(args.output_mode == "fixed"),
    )

    # custom -> chat() applies the model's own chat template (special tokens, and
    # for hybrid-reasoning models the enable_thinking switch); sharegpt -> send the
    # raw prompt through generate().
    t0 = time.perf_counter()
    if args.dataset_kind == "custom":
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        # enable_thinking is Qwen3-specific: only forward it when the model's chat
        # template actually references it, so Llama etc. don't get an unused kwarg.
        chat_kwargs = {}
        template = getattr(llm.get_tokenizer(), "chat_template", None) or ""
        if "enable_thinking" in template:
            chat_kwargs["enable_thinking"] = args.enable_thinking
        outs = llm.chat(conversations, sampling, chat_template_kwargs=chat_kwargs)
    else:
        outs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    prompt_lens = [len(o.prompt_token_ids) for o in outs]
    output_lens = [len(o.outputs[0].token_ids) for o in outs if o.outputs]
    total_in = sum(prompt_lens)
    total_out = sum(output_lens)
    total = total_in + total_out
    # Output cap only censors natural mode (fixed mode forces exactly output_len,
    # so "clipped" is meaningless there).
    out_cap = args.output_len if args.output_mode == "natural" else None
    res = {
        "config": args.config,
        "max_num_seqs": args.max_num_seqs,
        "num_prompts": len(prompts),
        "elapsed_time": elapsed,
        "total_prompt_tokens": total_in,
        "total_output_tokens": total_out,
        "total_num_tokens": total,
        "requests_per_second": len(prompts) / elapsed,
        "output_tokens_per_second": total_out / elapsed,
        "total_tokens_per_second": total / elapsed,
        "avg_output_len": total_out / len(prompts),
        "prompt_len_stats": len_stats(prompt_lens),
        "output_len_stats": len_stats(output_lens, cap=out_cap),
        "temperature": args.temperature,
        "output_len": args.output_len,
        "output_mode": args.output_mode,
        "prefix_caching": args.prefix_caching,
        "spec": args.spec or "",
    }
    if args.save_lengths:  # raw arrays for histograms (off by default: ~2*N ints)
        res["prompt_lens"] = prompt_lens
        res["output_lens"] = output_lens
    if args.save_detailed:  # per-request timings for steady-state/concurrency reconstruction
        res["detailed"] = _detailed_timings(outs)
    res.update(prefix_cache_metrics(llm))
    if spec:
        res.update(acceptance_metrics(llm, int(spec.get("num_speculative_tokens", 0))))
    if use_tp_sched and step_stats_path.exists():
        try:
            res["step_stats"] = json.loads(step_stats_path.read_text())
            ss = res["step_stats"]
            print(f"step_stats: steps={ss.get('n_steps')} "
                  f"pure_prefill={ss.get('n_pure_prefill')} "
                  f"pure_decode={ss.get('n_pure_decode')} "
                  f"mixed={ss.get('n_mixed')} ", flush=True)
        except Exception as e:
            print(f"step_stats: could not read {step_stats_path}: {e}", flush=True)
    Path(args.out_json).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
