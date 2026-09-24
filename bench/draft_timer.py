"""Time each engine step, and the EAGLE draft phase inside it, with CUDA events.

Patches vLLM 0.23's V2 runner: GPUModelRunner.execute_model and .sample_tokens
bracket the step (propose() is called from the second, not the first),
AutoRegressiveSpeculator.propose isolates the draft phase, and the CUDA-graph
managers catch the target forward. Overhead is then step - draft - target_fwd.

Installed either directly from tp_worker.py before LLM() (the engine is forked, so
it inherits the patch) or through a vllm.general_plugins entry point (spawned).
Events are drained with Event.query(), so the steady state never synchronizes.
Writes <out>_draft_pid<N>.csv; enable with SD_DRAFT_TIMER_PATH.
"""
import atexit
import csv
import io
import os
import signal
from pathlib import Path

ENV_PATH = "SD_DRAFT_TIMER_PATH"   # deliberately not VLLM_*: vLLM warns on
                                   # unknown VLLM_ vars and filters them on Ray.
FLUSH_EVERY = 200         # steps between non-blocking drain attempts
MAX_STEPS = 500_000       # hard cap so a runaway run cannot exhaust memory

FIELDS = ["step", "sched_reqs", "sched_tokens", "batch", "num_query_tokens",
          "sum_seq_len", "num_spec_tokens", "draft_ms", "target_fwd_ms",
          "n_target_fwd", "exec_ms", "sample_ms", "step_ms"]

_warned: set = set()
_base_path = None
_patched = False
_step_patched = False
_timers: dict = {}        # pid -> _Timer (a fork gets its own)


class _Timer:
    def __init__(self, path: Path):
        self.path = path
        self.pending: list = []
        self.step = 0
        self.header_done = False
        self.slot = None      # draft record for the step currently executing
        self.fwd: list = []   # target-forward event pairs for this step
        self.in_draft = False # True while inside propose(), to attribute forwards
        self.exec = None      # execute_model half, held until sample_tokens ends

    # -- collection ---------------------------------------------------------
    def stash_draft(self, rec) -> None:
        """propose() ran; hold it until execute_model closes the step."""
        self.slot = rec

    def record_exec(self, sched_reqs, sched_tokens, start, end) -> None:
        """First half of the step. Held until sample_tokens closes it."""
        fwd, self.fwd = self.fwd, []
        self.exec = (sched_reqs, sched_tokens, fwd, start, end)

    def record_step(self, start, end) -> None:
        """Second half (sample_tokens) -- emits the combined row."""
        if self.exec is None:
            self.slot = None
            return
        sreqs, stoks, fwd, es, ee = self.exec
        self.exec = None
        draft, self.slot = self.slot, None
        self.pending.append(
            (self.step, sreqs, stoks, draft, fwd, es, ee, start, end))
        self.step += 1
        if len(self.pending) >= FLUSH_EVERY:
            self.flush()

    def record_draft_alone(self) -> None:
        """Fallback: execute_model was not patched, so emit the draft on its own."""
        draft, self.slot = self.slot, None
        if draft is None:
            return
        self.pending.append((self.step, -1, -1, draft, [], None, None, None, None))
        self.step += 1
        if len(self.pending) >= FLUSH_EVERY:
            self.flush()

    # -- output -------------------------------------------------------------
    @staticmethod
    def _seq_len(sl):
        if sl is None:
            return -1
        return int(sl) if isinstance(sl, int) else int(sl.item())

    def _row(self, step, sreqs, stoks, draft, fwd, es, ee, ss, se):
        r = dict.fromkeys(FIELDS, -1)
        r["step"], r["sched_reqs"], r["sched_tokens"] = step, sreqs, stoks
        ex = es.elapsed_time(ee) if es is not None else None
        sm = ss.elapsed_time(se) if ss is not None else None
        if ex is not None:
            r["exec_ms"] = round(ex, 6)
        if sm is not None:
            r["sample_ms"] = round(sm, 6)
        if ex is not None and sm is not None:
            r["step_ms"] = round(ex + sm, 6)
        r["n_target_fwd"] = len(fwd)
        if fwd:
            r["target_fwd_ms"] = round(sum(a.elapsed_time(b) for a, b in fwd), 6)
        if draft is not None:
            b, ntok, sl, nspec, ds, de = draft
            r.update(batch=b, num_query_tokens=ntok, sum_seq_len=self._seq_len(sl),
                     num_spec_tokens=nspec, draft_ms=round(ds.elapsed_time(de), 6))
        return r

    def flush(self, force: bool = False) -> None:
        if not self.pending:
            return
        try:
            import torch

            if force:
                torch.cuda.synchronize()
                ready, self.pending = self.pending, []
            else:
                n = 0
                for rec in self.pending:
                    last = rec[8] if rec[8] is not None else (
                        rec[3][5] if rec[3] is not None else None)
                    if last is None or not last.query():
                        break
                    n += 1
                if not n:
                    return
                ready, self.pending = self.pending[:n], self.pending[n:]

            rows = [self._row(*rec) for rec in ready]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=FIELDS)
            if not self.header_done and (
                not self.path.exists() or self.path.stat().st_size == 0
            ):
                w.writeheader()
            w.writerows(rows)
            with self.path.open("a", newline="") as f:
                f.write(buf.getvalue())
            self.header_done = True
        except Exception as exc:  # never take the engine down over telemetry
            self.pending.clear()
            print(f"[draft_timer] flush failed: {exc}", flush=True)


def _timer_for_pid():
    """One timer per process. A forked child builds its own on first use."""
    pid = os.getpid()
    t = _timers.get(pid)
    if t is None:
        stem = Path(_base_path)
        t = _Timer(stem.with_name(f"{stem.stem}_pid{pid}{stem.suffix or '.csv'}"))
        _timers[pid] = t
        atexit.register(lambda: t.flush(force=True))
        try:
            prev = signal.getsignal(signal.SIGTERM)

            def _on_term(signum, frame):
                t.flush(force=True)
                if callable(prev):
                    prev(signum, frame)
                elif prev == signal.SIG_DFL:
                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)

            signal.signal(signal.SIGTERM, _on_term)
        except Exception:
            pass
        print(f"[draft_timer] recording -> {t.path}", flush=True)
    return t


def _draft_covariates(obj, args, kwargs):
    """(batch, num_tokens, sum_seq_len, D) from propose()'s InputBatch."""
    ib = kwargs.get("input_batch")
    if ib is None and args:
        ib = args[0]
    n = int(ib.num_reqs)
    ub = getattr(ib, "seq_lens_cpu_upper_bound", None)
    sl = int(ub[:n].sum().item()) if ub is not None else -1
    return (n, int(getattr(ib, "num_tokens", -1) or -1), sl,
            int(getattr(obj, "num_speculative_steps", -1)))


def _step_covariates(args, kwargs):
    """(num_reqs, num_scheduled_tokens, is_dummy) from a SchedulerOutput."""
    so = kwargs.get("scheduler_output")
    if so is None and args:
        so = args[0]
    dummy = bool(kwargs.get("dummy_run", False) or kwargs.get("is_profile", False)
                 or (len(args) >= 3 and args[2]))
    ns = getattr(so, "num_scheduled_tokens", None)
    return (len(ns) if ns is not None else -1,
            int(getattr(so, "total_num_scheduled_tokens", -1) or -1),
            dummy)


def install() -> None:
    """Patch propose() and execute_model(). Idempotent (plugins load per process)."""
    global _patched, _step_patched, _base_path
    if _patched:
        return
    raw = os.environ.get(ENV_PATH, "").strip()
    if not raw:
        return
    try:
        import torch
    except Exception as exc:
        print(f"[draft_timer] not installed: {exc}", flush=True)
        return
    _base_path = raw

    def _wrap_propose(orig):
        def propose(self, *args, **kwargs):
            timer = _timer_for_pid()
            if timer.step >= MAX_STEPS:
                return orig(self, *args, **kwargs)
            try:
                b, ntok, sl, nspec = _draft_covariates(self, args, kwargs)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
            except Exception as exc:
                # Silent failure here once cost a whole tier of draft data, so
                # say it out loud (once per process).
                if "propose" not in _warned:
                    _warned.add("propose")
                    print(f"[draft_timer] propose covariates failed, draft will "
                          f"NOT be timed: {type(exc).__name__}: {exc}", flush=True)
                return orig(self, *args, **kwargs)
            s.record()
            timer.in_draft = True
            try:
                return orig(self, *args, **kwargs)
            finally:
                timer.in_draft = False
                e.record()
                try:
                    timer.stash_draft((b, ntok, sl, nspec, s, e))
                    if not _step_patched:
                        timer.record_draft_alone()
                except Exception:
                    pass
        return propose

    def _wrap_forward(orig):
        """Time one target-model forward. Draft forwards go through the same
        graph manager, so only count it when not inside propose()."""
        def run(self, *args, **kwargs):
            timer = _timers.get(os.getpid())
            if timer is None or timer.in_draft or timer.step >= MAX_STEPS:
                return orig(self, *args, **kwargs)
            try:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
            except Exception:
                return orig(self, *args, **kwargs)
            s.record()
            try:
                return orig(self, *args, **kwargs)
            finally:
                e.record()
                try:
                    timer.fwd.append((s, e))
                except Exception:
                    pass
        return run

    def _wrap_sample(orig):
        """Second half of an engine step. propose() is called from HERE, not
        from execute_model, so this is where the row gets closed."""
        def sample_tokens(self, *args, **kwargs):
            timer = _timers.get(os.getpid())
            if timer is None or timer.exec is None or timer.step >= MAX_STEPS:
                return orig(self, *args, **kwargs)
            try:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
            except Exception:
                return orig(self, *args, **kwargs)
            s.record()
            try:
                return orig(self, *args, **kwargs)
            finally:
                e.record()
                try:
                    timer.record_step(s, e)
                except Exception:
                    pass
        return sample_tokens

    def _wrap_step(orig):
        def execute_model(self, *args, **kwargs):
            timer = _timer_for_pid()
            if timer.step >= MAX_STEPS:
                return orig(self, *args, **kwargs)
            timer.slot = None
            timer.fwd = []
            try:
                sreqs, stoks, dummy = _step_covariates(args, kwargs)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
            except Exception as exc:
                if "step" not in _warned:
                    _warned.add("step")
                    print(f"[draft_timer] step covariates failed, step will NOT "
                          f"be timed: {type(exc).__name__}: {exc}", flush=True)
                return orig(self, *args, **kwargs)
            s.record()
            try:
                return orig(self, *args, **kwargs)
            finally:
                e.record()
                try:
                    if dummy:
                        timer.slot = None      # drop warmup/profile draft too
                        timer.fwd = []
                        timer.exec = None
                    else:
                        timer.record_exec(sreqs, stoks, s, e)
                except Exception:
                    pass
        return execute_model

    targets = [
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner",
         "execute_model", _wrap_step),
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner",
         "sample_tokens", _wrap_sample),
        ("vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
         "AutoRegressiveSpeculator", "propose", _wrap_propose),
        # Only the SUBCLASS run_fullgraph: it calls super(), so patching the
        # base as well would count every FULL replay twice.
        ("vllm.v1.worker.gpu.cudagraph_utils", "ModelCudaGraphManager",
         "run_fullgraph", _wrap_forward),
        ("vllm.v1.worker.gpu.cudagraph_utils", "CudaGraphManager",
         "run_pw_graph", _wrap_forward),
    ]
    hit = []
    for mod_path, cls_name, meth, wrapper in targets:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            fn = getattr(cls, meth)
            if getattr(fn, "_sd_wrapped", False):
                continue
            new = wrapper(fn)
            new._sd_wrapped = True
            setattr(cls, meth, new)
            hit.append(f"{cls_name}.{meth}")
            if meth == "execute_model":
                _step_patched = True
        except Exception as exc:
            print(f"[draft_timer] skip {cls_name}.{meth}: {exc}", flush=True)
    if not hit:
        print("[draft_timer] NOTHING patched -- no rows will be recorded", flush=True)
        return
    _patched = True
    print(f"[draft_timer] patched {', '.join(hit)} in pid {os.getpid()} -> {raw}",
          flush=True)


def register() -> None:
    """vllm.general_plugins entry point."""
    install()
