"""D2D link contention model for disaggregated LLM serving.

Ports run.py's D2DState + KVQueue + run_kv_until + reserve_decode_collective
to iteration-level granularity.

Single-server model: D2D link is serialized; KV transfers and decode
collectives compete for it. KV runs opportunistically during compute phases;
decode collectives are explicitly reserved at compute-phase end.

Units: nanoseconds (ns), bytes, GB/s (= bytes/ns numerically).
"""

from __future__ import annotations

import heapq
import math
from typing import Optional


class D2DContentionModel:
    """Single-server D2D link contention tracker.

    Mirrors run.py:
      stall_kv        -> KV ready but D2D busy with collective
      stall_collective -> collective wants D2D but it's busy with KV
      stall_total = stall_kv + stall_collective  (NOT 2x overlap; genuinely different values)
    """

    def __init__(self, link_bw_gbps: float, link_latency_ns: float) -> None:
        self._bw    = link_bw_gbps    # bytes/ns  (1 GB/s = 1 byte/ns)
        self._alpha = link_latency_ns # ns  (channel_latency_decode_us equivalent)

        # D2D link state (mirrors D2DState in run.py)
        self._d2d_avail: int = 0
        self._last_kind: Optional[str] = None   # 'kv' | 'collective' | None

        # KV job queue (mirrors KVQueue in run.py): min-heap (ready_ns, seq, dur_ns)
        self._kv_heap: list = []
        self._seq: int = 0

        # Accumulated stall counters
        self.stall_kv:          int = 0   # ns blocked by collective
        self.stall_collective:  int = 0   # ns blocked by KV

        # Diagnostics
        self._n_kv_jobs:   int = 0
        self._n_coll_ops:  int = 0

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def add_kv_transfer(self, ready_ns: int, kv_bytes: int) -> None:
        """Register a 2-hop KV transfer available at ready_ns.

        kv_bytes: from memory.get_total_kv(req)  [bytes, per TP rank]
        duration = 2 x kv_bytes / link_bw   (2-hop model from run.py)
        """
        if kv_bytes <= 0 or self._bw <= 0:
            return
        dur_ns = int(math.ceil(2 * kv_bytes / self._bw))
        self._seq += 1
        heapq.heappush(self._kv_heap, (ready_ns, self._seq, dur_ns))
        self._n_kv_jobs += 1

    def add_decode_collective(
        self,
        iter_start_ns: int,
        iter_end_ns:   int,
        hidden_size:   int,
        num_layers:    int,
        tp_size:       int,
        total_len:     int,
        fp_bits:       int,
    ) -> None:
        """Model one decode iteration's ALLREDUCE window.

        Mirrors one token step in run.py's simulate_decode_for_batch_1_tok:
          1. KV runs opportunistically during the compute phase (before collective)
          2. Collective explicitly reserves D2D (may be stalled by in-flight KV)

        collective_want_ns = iter_end_ns - collective_dur_ns
          -> end-anchored: ASTRA-Sim runs collective last in each layer's trace
        """
        if tp_size <= 1:
            return   # no D2D collective; no contention from this side

        collective_dur_ns = self._collective_dur_ns(
            hidden_size, num_layers, tp_size, total_len, fp_bits
        )
        # Collective "wants" D2D at the end of the compute phase
        collective_want_ns = max(iter_start_ns,
                                 iter_end_ns - int(collective_dur_ns))

        # Phase 1: let KV run during the compute window (mirrors run_kv_until)
        self._run_kv_until(collective_want_ns)

        # Phase 2: reserve D2D for the collective (mirrors reserve_decode_collective)
        actual_start = max(self._d2d_avail, collective_want_ns)
        if collective_want_ns < self._d2d_avail and self._last_kind == 'kv':
            self.stall_collective += self._d2d_avail - collective_want_ns

        self._d2d_avail = actual_start + int(collective_dur_ns)
        self._last_kind = 'collective'
        self._n_coll_ops += 1

    # ------------------------------------------------------------------
    # Final stall computation
    # ------------------------------------------------------------------

    def compute_stalls(self, total_sim_ns: int) -> dict:
        """Drain remaining KV queue and return stall metrics."""
        # Drain: process any KV jobs that were never given a compute window
        self._run_kv_until(int(1e18))

        stall_total = self.stall_kv + self.stall_collective
        ratio = stall_total / total_sim_ns if total_sim_ns > 0 else 0.0
        return {
            "stall_kv_due_to_decode_collective_ns":  self.stall_kv,
            "stall_decode_collective_due_to_kv_ns":  self.stall_collective,
            "stall_total_ns":                         stall_total,
            "stall_ratio":                            ratio,
            "num_kv_jobs":                            self._n_kv_jobs,
            "num_decode_collective_ops":              self._n_coll_ops,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_kv_until(self, t_limit_ns: int) -> None:
        """Execute KV jobs whose ready_t <= t_limit_ns, if D2D is free by then.

        Mirrors run_kv_until() in run.py exactly:
          - Job must be ready (ready_t <= t_limit)
          - Job must be startable before t_limit (start = max(avail, ready) < t_limit)
          - Stall counted when KV is ready but D2D is busy with a collective
        """
        while self._kv_heap:
            ready_ns, _, dur_ns = self._kv_heap[0]
            if ready_ns > t_limit_ns:
                break
            start = max(self._d2d_avail, ready_ns)
            if start >= t_limit_ns:
                break   # D2D won't free before t_limit; job stays in queue
            # Stall: KV ready but D2D occupied by a collective
            if ready_ns < self._d2d_avail and self._last_kind == 'collective':
                self.stall_kv += self._d2d_avail - ready_ns
            heapq.heappop(self._kv_heap)
            self._d2d_avail = start + dur_ns
            self._last_kind = 'kv'

    def _collective_dur_ns(
        self,
        hidden_size: int,
        num_layers: int,
        tp_size: int,
        total_len: int,
        fp_bits: int,
    ) -> float:
        """Ring ALLREDUCE duration for attn + ffn, all layers.

        Formula from metrics_engine.py:
          t_one = 2*(N-1)*(alpha + msg_bytes/(N*BW))
          msg_bytes = total_len * H * fp_bytes
          total = num_layers * 2 * t_one   (attn ALLREDUCE + ffn ALLREDUCE per layer)
        """
        fp_bytes  = fp_bits // 8                               # e.g. 16 bits -> 2 bytes
        msg_bytes = total_len * hidden_size * fp_bytes
        t_one_ns  = 2 * (tp_size - 1) * (
            self._alpha + msg_bytes / (tp_size * self._bw)
        )
        return num_layers * 2 * t_one_ns
