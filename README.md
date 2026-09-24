# When Does Speculative Decoding Pay Off?

Below is a roofline model of the decode pass that predicts when speculative decoding stops helping, and a vLLM benchmark that tests the prediction. We optimized for the **in-database operating point** using our model: offline serving, throughput-maximizing, no latency SLA, with the batch pinned to whatever the KV cache will hold. We then tested this configuration in vLLM.

Research done under Prof. Jignesh Patel at CMU, on in-database LLM query workloads.

---

## Finding

Speculative decoding spends compute, which is free while the pass is weight-bound, to avoid weight loads. In-DB queries have long inputs and short outputs, which pushes achievable batch past the point where the verify pass is weight-bound.

**Metrics:** Throughput is **output tokens/s per GPU**: `out_tok/s` divided by TP. Measurements are **steady-state** (`ss_out_tok/s`). Overall `out_tok/s` is diluted by ramp-up and drain, which run at lower batch and thus favor speculation.

**Key result:** Llama-3.1-8B + EAGLE-3 on an H200 at fp8, the configuration the model picks for maximum throughput, with each arm at its best batch (details in *Experiments*):

| workload | measured | model at measured $\Omega$ | model at α=0.80 |
|---|---|---|---|
| relqueries | **−8.0%** | −10.0% | −8.8% |
| PDMX | **−5.0%** | −4.1% | +0.9% |

Speculation loses on both DB workloads at the batch that maximizes throughput, and the model at measured acceptance is within 2 points of each.

---

## Model

From [`roofline/derivation_notes_adjusted/`](roofline/derivation_notes_adjusted/). Symbols:

| symbol | meaning |
|---|---|
| $C$, $\beta$ | accelerator compute (FLOP/s) and memory bandwidth (bytes/s) |
| $M_{HBM}$, $M_{res}$ | HBM capacity and the headroom held back from the KV cache |
| $TP$ | tensor-parallel size |
| $N$, $b_w$ | target parameter count and bytes per weight |
| $n_{layers}$, $n_q$, $n_{kv}$, $d_h$ | layers, query heads, KV heads, head dimension |
| $b_{kv}$ | bytes per cached KV element |
| $B$ | batch, in concurrent sequences |
| $L$, $L_{in}$, $L_{out}$ | context, prompt, and generated tokens; $L_{avg} = L_{in} + L_{out}/2$, and $L_{max}$ is the prompt plus the maximum output |
| $\kappa$, $a$ | KV bytes per token per sequence, and attention seconds per query token per token of context |
| $w = \kappa/\beta$ | KV read seconds per token per sequence |
| $D$, $V$, $\Omega$ | draft passes per round, drafted positions ($V{+}1$ verified), and accepted tokens per round |
| $h$ | prefix cache hit rate |
| subscript $d$ | the drafter: $N_d$, $t_{c,d}$, $\kappa_{d,eff}$ and so on, formed the same way |

Times are seconds and capacities bytes. $t_{kv}$ is per sequence, $t_{attn}$ per query token.

**Primitives:** Per pass, whole model:

$$
t_w = \frac{N b_w}{\beta}
\qquad
t_c = \frac{2N}{C}
\qquad
t_{kv}(L) = \frac{\kappa L}{\beta}
\qquad
t_{attn}(L) = aL
$$

$$
\kappa = 2\, n_{layers}\, n_{kv}\, d_h\, b_{kv}
\qquad\qquad
a = \frac{4\, n_{layers}\, n_q\, d_h}{C}
$$

$t_w$ is the weight load; $t_c$ is compute per token pushed through; $t_{kv}$ is the KV read per sequence. $t_{attn}$ is per query token at context $L$. $QK^\top$ and $AV$ are each $n_q d_h$ multiply-accumulates per layer per context token, and a one-layer EAGLE-3 head has $a_d = a / n_{layers}$.

**Prefill:**

$$
T_{pre}^{AR} = \max\left(B\left(t_c L_{in} + \tfrac{1}{2} a L_{in}^2\right)(1-h),\ t_w\right) + B\, t_{kv}(L_{in})(1-h)
$$

$$
T_{pre}^{spec} = T_{pre}^{AR} + \max\left(B\left(t_{c,d} L_{in} + \tfrac{1}{2} a_d L_{in}^2\right)(1-h),\ t_{w,d}\right) + B\, t_{kv,d}(L_{in})(1-h)
$$

The drafter pays a full prefill: an EAGLE-3 head reads the target's hidden states, so it cannot skip the prompt.

**Decode:** $D$ draft passes, then one verify pass over $V+1$ positions yielding $\Omega$ accepted tokens

$$
T_{dec}^{AR}(L) = \max\left(B\left(t_c + t_{attn}(L)\right),\ t_w\right) + B\, t_{kv}(L+1)
$$

$$
T_{verify}(L) = \max\left((V{+}1)\, B\left(t_c + t_{attn}(L)\right),\ t_w\right) + B\, t_{kv}(L{+}V{+}1)
$$

$$
T_{draft}(L) = D\left[\max\left(B\left(t_{c,d} + t_{attn,d}(L)\right),\ t_{w,d}\right) + B\, t_{kv,d}(L)\right]
$$

$$
T_{dec}^{spec}(L) = T_{verify}(L) + T_{draft}(L)
$$

**End-to-end speedup, and batch capacity:**

$$
S_{e2e} = \frac{T_{pre}^{AR} + L_{out}\, T_{dec}^{AR}(L_{avg})}
               {T_{pre}^{spec} + \dfrac{L_{out}}{\Omega}\, T_{dec}^{spec}(L_{avg})}
$$

$$
B_{KV} = \frac{TP\left(M_{HBM} - M_{res}\right) - N b_w - N_d b_{w,d}}
              {\left(\kappa_{eff} + \kappa_{d,eff}\right) L_{max}}
\qquad
\kappa_{eff} = \kappa \max\left(1, \tfrac{TP}{n_{kv}}\right)
$$

**Break-even batch:** The decode-only speedup is $S_{dec} = \Omega\, T_{dec}^{AR} / T_{dec}^{spec}$, which is what $B_{be}$ and $L_{crit}$ are built on, not $S_{e2e}$. Setting $S_{dec} = 1$ on the branch where the AR pass and the drafter are weight-bound and the verify pass is compute-bound, taking $L+V+1 \approx L$:

$$
B_{be} = \frac{\Omega\, t_w - D\, t_{w,d}}{(V{+}1)\left(t_c + t_{attn}(L)\right) - (\Omega-1)\, t_{kv}(L) + D\, t_{kv,d}(L)}
$$

The KV rebate $(\Omega-1)\, t_{kv}(L)$ grows linearly in $L$ and shrinks the denominator. At a saturated batch every pass is compute-bound; dividing through by $B$ and solving $S_{dec}=1$ for $L$:

$$
L_{crit} = \frac{(V{+}1-\Omega)\, t_c + D\, t_{c,d}}{(\Omega-1)\, w - (V{+}1-\Omega)\, a - D\,(a_d + w_d)}
$$

Below $L_{crit}$, $B_{be}$ is finite. At or above it the model says speculation wins at every batch and $B_{be} = \infty$. For Llama-3.1-8B + EAGLE-3 on A100-40 at depth 3 and $\Omega = 2.44$, for example, $L_{crit} \approx 800$ and $B_{be}(576) = 163$.

For $L < L_{crit}$ there is no reason to test past $B_{be}$: the model already predicts a loss there, and it is optimistic about where that loss starts (see *Model gaps*). Offline serving runs at $B_{KV}$, so any workload with $B_{KV} > B_{be}$ is in that region at its operating point. The converse does not hold above $L_{crit}$: $B_{be} = \infty$ comes from the over-credited KV rebate and is not a guarantee.

### Assumptions

Each is listed with the direction of its error.

- Vendor $C$ and $\beta$ are achieved: **optimistic**
- Weight streaming and compute overlap perfectly: **optimistic**
- $\Omega$ is asymptotic and independent of $B$ and $L_{in}$: **optimistic**
- Draft overhead beyond the modelled passes is zero: **optimistic**
- TP all-reduce is priced at zero: **pessimistic**, since it is a per-pass fixed cost that speculation amortizes over $V{+}1$ positions
- Time primitives are one-GPU-equivalent: **optimistic for TP>1**
- Prefill and decode are priced as separate passes: **pessimistic**. With chunked prefill, vLLM runs most decode inside mixed prefill-and-decode steps (about 90% of speculative step time), where the verify positions share compute the prefill already pays for

---

## Model error

The model is optimistic on speedup. We test across workloads using GovReport (`ccdv/govreport-summarization`): U.S. government reports truncated to 512, 1k, or 2k input tokens, with an instruction steering the summary toward 32, 128, or 512 output tokens. Generation stops at EOS, every input tier draws from the same documents, and each run is 1024 prompts, on Llama-3.1-8B + EAGLE-3, A100-40, TP=1, with depth 3.

| workload | B | Ω | measured | model | error |
|---|---|---|---|---|---|
| 512/32 | 32 | 2.24 | 1.11x | 1.12x | +1% |
| 512/32 | 128 | 2.25 | 0.97x | 0.96x | −1% |
| 1k/32 | 32 | 2.25 | 1.02x | 1.05x | +3% |
| 1k/32 | 128 | 2.25 | 0.98x | 0.97x | −1% |
| 2k/32 | 32 | 2.24 | 1.00x | 1.01x | +1% |
| 2k/32 | 64 | 2.23 | 1.01x | 0.99x | −2% |
| 512/128 | 32 | 2.43 | 1.34x | 1.46x | +9% |
| 1k/128 | 32 | 2.42 | 1.22x | 1.30x | +7% |
| 1k/128 | 128 | 2.41 | 1.01x | 1.02x | +1% |
| 2k/128 | 32 | 2.41 | 1.14x | 1.19x | +4% |
| 512/512 | 32 | 2.63 | 1.60x | 1.92x | +20% |
| 1k/512 | 32 | 2.70 | 1.43x | 1.80x | +26% |
| 2k/512 | 64 | 2.68 | 1.14x | 1.42x | +24% |

The regime where the model is most accurate is the regime in-DB workloads live in. At $L_{out}$=32 the error is −2% to +3%. The +20 to 26% errors are all at $L_{out}$=512, which is beyond in-DB workload outputs, as far as we know.

---

## Optimization setup

What happens if we optimize the model given different constraints?

Search space:

| axis | values |
|---|---|
| hardware | A100-40, A100-80, H100-80, H200, L40S |
| dtype | bf16, fp8 (fp8 on A100 has no FP8 unit, so it is weight-only and marked `*`) |
| target | Llama-3.1-8B, Qwen3-8B, Qwen3-14B, Qwen3-32B, Llama-3.3-70B |
| tensor parallel | 1, 2, 4, 8 |
| chain depth | 1, 2, 3, 4, 5 |
| batch | 8 … 512, capped at $B_{KV}$ for that arm |
| workload | fixed per row; $L_{in}$ ≤ 2048, $L_{out}$ ≤ 256 |

Fixed variables:

- Acceptance α = 0.80 per position, above what we measured
- Every target is one with a public EAGLE-3 head, so the winner is runnable
- Draft heads use their actual checkpoint sizes (0.85 GB for Llama-3.1-8B, 2.0 to 3.2 GB for the others) and stay bf16 when the target is fp8, which is how vLLM loads them
- The AR arm carries neither the drafter's weights nor its KV cache, so it gets the larger $B_{KV}$

## Optimization 1: maximize throughput

An important question is: what if we optimize for throughput? We test this at the edges of feasible target models, 8B and 70B. Speculation enters as a yes/no decision alongside hardware, dtype, TP, depth, and batch. We consider four workloads exploring each extreme; short extraction or classification (relqueries), a short summary (PDMX), a long summary, and a long generation from a short input. 

**Llama-3.1-8B**

| workload | winner | B | speculate? | tok/s/GPU | delta |
|---|---|---|---|---|---|
| relqueries 80/10 | H200 fp8 TP1 | 256 | no | 13208 | −8.8% if forced |
| PDMX 887/71 | H200 fp8 TP1 | 256 | yes, depth 3, S=1.01x | 7377 | +0.9% |
| long summary 2048/256 | H200 fp8 TP1 | 384 | yes, depth 4, S=1.25x | 9246 | +25.2% |
| long generation 256/256 | H200 fp8 TP1 | 256 | yes, depth 2, S=1.10x | 40905 | +10.1% |

**Llama-3.3-70B**

| workload | winner | B | speculate? | tok/s/GPU | delta |
|---|---|---|---|---|---|
| relqueries 80/10 | H200 fp8 TP1 | 256 | no | 1541 | −4.8% if forced |
| PDMX 887/71 | H200 fp8 TP2 | 256 | no | 966 | −2.1% if forced |
| long summary 2048/256 | H200 fp8 TP4 | 384 | yes, depth 3, S=1.05x | 1309 | +5.4% |
| long generation 256/256 | H200 fp8 TP1 | 256 | no | 5892 | −0.9% if forced |

Extraction and classification never benefit. For the 8B, speculation matters only once outputs are long (+10% to +25%), and PDMX's short summaries gain under 1%. The 70B speculates in one workload of four configurations for 5% gain. Per token its compute is 8.8x the 8B's while its KV read is only 2.5x, so the verify pass's extra compute outweighs the memory reads speculation saves.

Notice that every winner is with the H200 at fp8 with a large batch. Throughput is won by hardware, precision, and batch across each, and speculation is almost a secondary term.

## Optimization 2: maximize speculative speedup

Now what if we look for a configuration where speculative decoding should win? Let's change the objective to $S_{e2e}$: first at each workload above, then with $L_{in}$ and $L_{out}$ also free variables.

| workload | best $S_{e2e}$ | tok/s/GPU | share of max throughput |
|---|---|---|---|
| relqueries 80/10 | 2.33x | 362 | 2.7% |
| PDMX 887/71 | 2.07x | 290 | 3.9% |
| long summary 2048/256 | 2.34x | 352 | 3.8% |
| long generation 256/256 | 3.12x | 573 | 1.4% |
| any length in the box: 80/256 | 3.25x | 610 | 1.0% |

Every row is the same configuration: H100-80, bf16, Llama-3.3-70B, TP4, batch 8, depth 5. Batch 8 is the smallest searched and depth 5 the deepest, so the model is bound by both extremes. At batch 8 the verify pass sits far below the ridge, where its extra tokens really are free, which is why the 70B wins here while losing at max throughput. Over all lengths the optimizer picks the shortest input and longest output, where decode is most of the runtime. This represents a workload that doesn't really exist in our setting.

**The speedup-optimal configuration delivers 1 to 4% of the throughput available at the same workload.** Small batches and large models, the opposite of what maximizes throughput, provide the greatest end to end speedup.

---

## Experiments

### Optimization 1's winners on the DB workloads

Llama-3.1-8B + EAGLE-3 on one H200 at fp8, the configuration Optimization 1 picks for both workloads, at the depth it picks for each. Both arms, batch 128, 256, and 384, 3 repeats per point. Throughput is steady-state tok/s/GPU, shown as the mean with the range across repeats. The model is evaluated at each run's measured $\Omega$.

**relqueries** (77 input / 9 output tokens, depth 1, 20,000 prompts, Ω=1.63). Run `llama8b_relqueries_tp_d1_fp8/20260914_215434`.

| B | baseline | speculative | measured speedup | model speedup |
|---|---|---|---|---|
| 128 | 3937 [3777–4024] | 4275 [4200–4368] | 1.09x | 0.95x |
| 256 | 5486 [5380–5566] | 5067 [5031–5116] | 0.92x | 0.90x |
| 384 | **5959** [5865–6016] | **5485** [5374–5618] | 0.92x | 0.90x |

**PDMX** (881 input / 73 output tokens, depth 3, 2,560 prompts, Ω=2.21). Run `llama8b_pdmx_tp_d3_fp8/20260914_153841`.

| B | baseline | speculative | measured speedup | model speedup |
|---|---|---|---|---|
| 128 | 4265 [4258–4272] | 4157 [4152–4166] | 0.97x | 0.99x |
| 256 | 4470 [4459–4475] | 4307 [4293–4317] | 0.96x | 0.96x |
| 384 | **4619** [4614–4623] | **4389** [4375–4397] | 0.95x | 0.96x |

The gap between Optimization 1's PDMX prediction and the measurement is almost entirely acceptance. We measure Ω=2.21 while we assumed Ω=2.95. On the relqueries experiment, speculation gains 9% at batch 128 and loses once the batch fills. The model reproduces this in direction, but is pessimistic at batch 128.

---

## Workload scope

The workload was restricted to what an in-database LLM query plausibly looks like and designed in speculative decoding's favor. For each setting we attempted to replicate the prompts from the literature.

| | $L_{in}$ | $L_{out}$ | source |
|---|---|---|---|
| relqueries | 79.5 | 9.8 | *RelServe*, filter/classify/summarize/QA templates over 4 datasets |
| PDMX | 886.7 | 70.6 | *Optimizing LLM Queries in Relational Data Analytics Workloads*, upper extreme |
| **search box** | **≤ 2048** | **≤ 256** | |

$L_{in}$ ≤ 2048: longer than this is at the size of a document, not a field, and existing literature uses datasets well within this range. For larger context, rows are sometimes trimmed to include only relevant columns. $L_{out}$ ≤ 256: summaries are the upper end of decode and seem to be below this length.

---

## Model gaps

Although the model's optimism is beneficial for making experimental decisions, I was curious why this was the case, so I patched vLLM's V2 model runner with CUDA-event instrumentation ([`bench/draft_timer.py`](bench/draft_timer.py)) to time each generation step.

`GPUModelRunner.execute_model` and `.sample_tokens` bracket the engine step, `AutoRegressiveSpeculator.propose` isolates the draft phase, and `ModelCudaGraphManager.run_fullgraph` / `CudaGraphManager.run_pw_graph` catch the target forward.

**Method:** Both arms are timed on GovReport `out128` at 512, 1k, and 2k input tokens (Llama-3.1-8B + EAGLE-3, A100-40, depth 3; runs pinned in [`roofline/decomp3.py`](roofline/decomp3.py)). Every engine step writes one row with its total time, the target forward, and the draft phase. From these:

- Only pure decode steps are kept: on the baseline, all B sequences decoding one token; with speculation, all B sequences verifying V+1 tokens. Steps that mix in a prefill chunk are dropped.
- Each quantity is the median over those steps, since the first steps of a run include CUDA graph capture and batch fill.
- Overhead is step time minus the target forward minus the draft phase.

| gap | model | measured |
|---|---|---|
| **1. Verify pass free when weight-bound** | verify = 1.00x an AR pass | **1.35–1.46x** at B ≤ 32; 2.81x vs 1.73 predicted at B=128 |
| **2. Peak bandwidth is not achieved** | peak β | AR forward runs 1.18–1.35x the weight-bound prediction |
| per-step overhead | n/a | 3–6% of a step |

**Gap 1:** Calculated as median verify forward over median AR forward at the same $B$ and context against the model's $T_{verify}/T_{AR}$. We measure 1.35 to 1.46x where the model says 1.00, and 2.81x against 1.73 at $B$=128.

**Gap 2:** Calculated as median AR forward against the model's $T_{AR}$: 18 to 35% slower. It applies to both speculation and normal  decoding, so it largely cancels in the speedup. Peak $C$ is untested.

**Per-step overhead:** Calculated as step time minus the target forward minus the draft phase: 0.89 to 1.12 ms on the baseline and 1.02 to 2.81 ms with speculation, which is 3 to 6% of a step either way. Scheduling, sampling, and rejection bookkeeping are not where the model's error is.

---

## Scope and further research

- It would be beneficial to test disaggreated prefill and decode to determine how much error results from ignoring continuous batching
- One highly specific workload sits outside our scope entirely: `LLM.SUMMARIZE_AGG` (Snowflake), 8k input and 1k output at the extreme. Both the model and the $L_{crit}$ screening rule put that firmly in speculation's favorable regime
- Only EAGLE-3 drafters were measured. An n-gram or lookup drafter has near-zero draft cost and would change the $T_{spec}$ draft term

## Layout

```
bench/       vLLM throughput harness, dataset prep, CUDA-event draft timer
roofline/    model.py (primitives, B_be, B_KV), optimize.py (config search), decomp3.py (three-way step decomposition), predict_phase1.py, derivation_notes_adjusted/
results/     one directory per run: summary.txt, config.json, and the CUDA-event CSVs
analysis/    per-run throughput and length summaries
data_creation/  dataset builders (relqueries, PDMX, GovReport ladder)
```
