"""Reproduce the caption_pipeline.py progress-bar starvation (DAY2 run, 2026-09-01).

The main loop in caption_pipeline.py (lines ~2790-2896) is:

    while True:
        if not args.skip_preprocess:
            bridge_produced()                  # <-- BLOCKS on api_in_q.put()
            ...
        try:
            res = result_q.get(timeout=0.2)    # <-- only reached when produced_q is EMPTY
        ...
        _update_pbar()                         # bar only moves here

bridge_produced() (line 2795) drains produced_q in a `while True` loop and does
a BLOCKING `api_in_q.put(ApiJob(...))`; api_in_q is bounded:
maxsize = api_workers * 2 (line 2707). With total=1702 clips, 60 workers and
maxsize=120, the queue saturates in the first ~2 minutes (60 in flight + 120
queued). From then on the main thread ping-pongs
get_nowait(produced_q) -> blocking put(api_in_q): it only escapes
bridge_produced() when produced_q is fully drained, which requires workers to
have PICKED UP essentially all 1702 jobs. Meanwhile completed results pile up
in result_q unconsumed, so the tqdm bar sits frozen (observed on the DAY2 run:
30/1702 for ~47 minutes, then a jump to ~1450 when produced_q finally emptied
at 23:26 - full.jsonl went 9.5KB -> 11MB in that instant).

This script replicates that loop 1:1 (same structure, same bounded-queue
sizing rule api_in_q = 2*workers; mock API latency + mock preprocess
production) and prints a timeline of "what the bar shows" vs "what actually
finished".

Variants:
  old -> faithful copy of the current main loop (shows the freeze)
  fix -> feeding moved to a dedicated feeder thread; the main loop only
         consumes results (bar tracks completions)

Usage:
    python _test/pbar_starvation_sim.py [total_clips] [api_workers]

Time is accelerated (one clip ~1.5s of sim vs ~1-3min real; one slice every
~8ms vs ~1.5s real ffmpeg) but the structural ratios are preserved:
production outpaces completion early on, so api_in_q saturates quickly.
"""

import queue
import sys
import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    """Verbatim copy of caption_pipeline.SlidingWindowRateLimiter (line 1413)."""

    def __init__(self, max_rpm: int):
        self.max_rpm = max(1, max_rpm)
        self._timestamps = deque()
        self._cond = threading.Condition(threading.Lock())

    def acquire(self) -> None:
        with self._cond:
            while True:
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] <= now - 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_rpm:
                    self._timestamps.append(now)
                    return
                wait = self._timestamps[0] + 60.0 - now + 0.05
                self._cond.wait(timeout=max(wait, 0.01))


class Counters:
    def __init__(self):
        self._lock = threading.Lock()
        self.done = 0        # results actually produced by workers

    def incr_done(self):
        with self._lock:
            self.done += 1

    def get_done(self):
        with self._lock:
            return self.done


def api_worker(in_q, out_q, limiter, clip_seconds, counters):
    """Mock of _api_worker: two rate-limited calls, then one result (like
    best_of_n=1 legacy flow: G1 + P2 infer)."""
    while True:
        job = in_q.get()
        if job is None:
            return
        limiter.acquire()               # G1
        time.sleep(clip_seconds * 0.6)
        limiter.acquire()               # P2 infer
        time.sleep(clip_seconds * 0.4)
        out_q.put(job)
        counters.incr_done()


def preprocess_producer(out_q, total, slice_seconds):
    """Mock of the preprocess pool: produces slices faster than API workers
    finish, exactly like 2 ffmpeg workers against 60 throttled API workers."""
    for i in range(1, total + 1):
        time.sleep(slice_seconds)
        out_q.put(i)


def run(sim: str, total: int, n_workers: int, max_rpm: int,
        clip_seconds: float, slice_seconds: float, report_every: float = 15.0):
    produced_q = queue.Queue()
    api_in_q = queue.Queue(maxsize=n_workers * 2)   # caption_pipeline.py:2707
    result_q = queue.Queue()
    counters = Counters()

    limiter = SlidingWindowRateLimiter(max_rpm)
    workers = [threading.Thread(target=api_worker,
                                args=(api_in_q, result_q, limiter, clip_seconds, counters),
                                daemon=True)
               for _ in range(n_workers)]

    api_fed = 0
    results_received = 0
    pp_done = False
    pp_expected = total
    pp_received = 0
    bar_value = 0                       # what tqdm would display right now
    samples = []                        # (t, bar, actual_done, api_in_q, produced_q)
    t_start = time.monotonic()

    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            samples.append((round(time.monotonic() - t_start), bar_value,
                            counters.get_done(), api_in_q.qsize(), produced_q.qsize()))
            stop.wait(report_every)

    threading.Thread(target=monitor, daemon=True).start()

    threading.Thread(target=preprocess_producer,
                     args=(produced_q, total, slice_seconds), daemon=True).start()
    for t in workers:
        t.start()

    if sim == "fix":
        # FIX: the api_in_q feeding happens on its own thread; the main loop
        # below is then free to service result_q continuously.
        def feeder():
            nonlocal api_fed, pp_done, pp_received
            for _ in range(total):
                pres = produced_q.get()
                api_in_q.put(pres)
                api_fed += 1
                pp_received += 1
                if pp_received >= pp_expected:
                    pp_done = True
        threading.Thread(target=feeder, daemon=True).start()

    def bridge_produced():
        """1:1 copy of caption_pipeline.bridge_produced (line 2795), including
        the blocking api_in_q.put() that starves the main loop."""
        nonlocal pp_received, pp_done, api_fed
        drained = 0
        while True:
            try:
                pres = produced_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            api_in_q.put(pres)          # <-- BLOCKS here while the queue is full
            api_fed += 1
        pp_received += drained
        if pp_received >= pp_expected:
            pp_done = True

    # ---- main loop, 1:1 shape of caption_pipeline.py:2842-2896 ----
    while True:
        if sim == "old":
            bridge_produced()           # main thread monopolized while produced_q non-empty
        now = time.monotonic()
        try:
            res = result_q.get(timeout=0.2)
        except queue.Empty:
            if pp_done and api_fed <= results_received:
                break
            if now - t_start > 600:
                print("TIMEOUT: sim did not finish in 600s")
                break
            continue
        results_received += 1
        bar_value = results_received    # tqdm.update(1) equivalent
        if pp_done and results_received >= api_fed:
            break

    stop.set()
    time.sleep(0.05)
    elapsed = time.monotonic() - t_start
    print(f"\n=== sim={sim}  total={total} api_workers={n_workers} "
          f"api_in_q_maxsize={n_workers * 2} ===")
    print(f"{'t(s)':>6} {'BAR':>7} {'actual done':>12} {'api_in_q':>9} {'produced_q':>11}")
    prev_bar = -1
    for t, bar, done, qsz, pqsz in samples:
        flag = ""
        if t > 30 and bar < done - 20:
            flag = "   <-- BAR STALE"
        print(f"{t:>6} {bar:>7} {done:>12} {qsz:>9} {pqsz:>11}{flag}")
        prev_bar = bar
    print(f"final: bar={bar_value} actual_done={counters.get_done()} "
          f"elapsed={elapsed:.0f}s  bar_stale_by={counters.get_done() - bar_value}")
    return samples


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 1702
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print("### OLD main-loop structure (blocking put on the main thread) ###")
    run("old", total, workers, max_rpm=96 * 60, clip_seconds=1.5, slice_seconds=0.008)
    print()
    print("### FIXED structure (feeding moved off the main thread) ###")
    run("fix", total, workers, max_rpm=96 * 60, clip_seconds=1.5, slice_seconds=0.008)
