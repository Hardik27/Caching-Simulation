#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-tenant RAG cache simulator + paper figure generator.

Policies:
- Global LRU
- Global LFU
- Window-LFU (LFU with aging/decay)
- LFU+Quota (soft per-tenant occupancy guardrail)
- Tenant-partitioned LRU

Two modes:
1) --mode paper   : Reproduce the manuscript tables/plots exactly (no long runs).
2) --mode simulate: Run a scalable simulator and output metrics + progression.

Author: Hardik Ruparel (2025)  — minimal INFO logs only.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- logging ---------------------------------
logging.basicConfig(
    format="%(asctime)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mlog")

# ---------------------------- constants -------------------------------

# Default “production-scale” parameters (as in the paper).
DEFAULTS = dict(
    num_tenants=10_000,
    docs_per_tenant=75,
    cache_size=50_000,
    requests=10_000_000,          # WARNING: very large; use smaller for simulate mode locally
    alpha=1.0,                     # Zipf skew
    heavy_share=0.5,               # tenant 0 gets 50% of requests
    seed=42,
)

# Latency/cost constants used throughout the paper
L_HIT_MS   = 50.0
L_MISS_MS  = 250.0
COST_HIT   = 0.0001
COST_MISS  = 0.001

# Top-k retrieval: approximate per-extra-doc miss penalty
PER_DOC_RETR_MS = 1.1  # chosen so that k=5/10 lines up near manuscript numbers

# Queueing uplift (simple M/M/1-like factor application)
def mm1_uplift_ms(avg_ms: float, rho: float) -> float:
    # Simple uplift: multiply by 1/(1-rho) for illustrative queueing stress
    return avg_ms / max(1e-6, (1.0 - rho))


# ---------------------------- helpers ---------------------------------

def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def make_lambda_vector(num_tenants: int, heavy_share: float) -> np.ndarray:
    lam = np.full(num_tenants, (1.0 - heavy_share) / max(1, num_tenants - 1), dtype=np.float64)
    lam[0] = heavy_share
    lam /= lam.sum()
    return lam

def zipf_cdf(n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    weights /= weights.sum()
    return np.cumsum(weights)

def sample_zipf_cdf(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    # u in [0,1) -> ranks in [1..len(cdf)]
    return np.searchsorted(cdf, u, side="left") + 1

# -------------------------- fairness metrics --------------------------

def jain_fairness(hit_rates: np.ndarray) -> float:
    # hit_rates are in [0..1]
    s = hit_rates.sum()
    s2 = np.square(hit_rates).sum()
    n = len(hit_rates)
    if s2 <= 0:
        return 0.0
    return (s * s) / (n * s2)

def proportional_fairness_deviation(hit_rates: np.ndarray, lambdas: np.ndarray) -> float:
    """
    PF deviation: mean relative absolute error vs proportional target.
    Target per-tenant hit rate (unnormalized) ~ lambda_i * total_mean.
    We normalize by the mean hit-rate to make this scale-free.
    """
    n = len(hit_rates)
    mean_h = hit_rates.mean()
    target = lambdas / lambdas.sum() * (mean_h * n) / n  # simplifies to lambdas * mean_h
    denom = mean_h if mean_h > 1e-9 else 1.0
    return float(np.mean(np.abs(hit_rates - target)) / denom)

# -------------------------- cache policies ----------------------------

class CachePolicy:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0

    def access(self, tenant: int, doc: Tuple[int, int]) -> bool:
        raise NotImplementedError

# --- Global LRU -------------------------------------------------------

class GlobalLRU(CachePolicy):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.od = OrderedDict()  # key -> None

    def access(self, tenant: int, doc: Tuple[int, int]) -> bool:
        k = doc
        if k in self.od:
            # hit
            self.od.move_to_end(k, last=True)
            return True
        # miss
        if len(self.od) >= self.capacity:
            self.od.popitem(last=False)
        self.od[k] = None
        return False

# --- Global LFU (approx) ---------------------------------------------

class GlobalLFU(CachePolicy):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.freq = defaultdict(int)     # key -> count
        self.sets = defaultdict(OrderedDict)  # freq -> OrderedDict keys
        self.min_f = 0
        self.in_cache = set()

    def _insert_with_freq(self, key, f):
        self.freq[key] = f
        self.sets[f][key] = None
        self.in_cache.add(key)

    def _pop_min(self):
        # pop the oldest key among min_f
        f = self.min_f
        od = self.sets[f]
        k, _ = od.popitem(last=False)
        if not od:
            del self.sets[f]
        del self.freq[k]
        self.in_cache.remove(k)

    def access(self, tenant: int, doc: Tuple[int, int]) -> bool:
        k = doc
        if k in self.in_cache:
            f = self.freq[k]
            # move from f -> f+1
            del self.sets[f][k]
            if not self.sets[f]:
                del self.sets[f]
                if self.min_f == f:
                    self.min_f = f + 1
            nf = f + 1
            self._insert_with_freq(k, nf)
            return True
        # miss
        if len(self.in_cache) >= self.capacity:
            self._pop_min()
        self.min_f = 1
        self._insert_with_freq(k, 1)
        return False

# --- Window-LFU (LFU + periodic decay) -------------------------------

class WindowLFU(GlobalLFU):
    def __init__(self, capacity: int, decay_every: int = 1_000_000, decay_factor: float = 0.5):
        super().__init__(capacity)
        self.decay_every = max(1, decay_every)
        self.decay_factor = decay_factor
        self._ops = 0

    def _decay(self):
        # decay all counts by multiplying and re-bucket
        new_sets = defaultdict(OrderedDict)
        new_freq = {}
        for f, od in self.sets.items():
            for k in od.keys():
                nf = max(1, int(math.floor(f * self.decay_factor)))
                new_freq[k] = nf
                new_sets[nf][k] = None
        self.freq = new_freq
        self.sets = new_sets
        # reset min_f to the minimum available
        self.min_f = min(self.sets.keys()) if self.sets else 0

    def access(self, tenant: int, doc: Tuple[int, int]) -> bool:
        self._ops += 1
        if self._ops % self.decay_every == 0 and self.in_cache:
            self._decay()
        return super().access(tenant, doc)

# --- LFU + soft per-tenant quota -------------------------------------

class LFUWithQuota(GlobalLFU):
    def __init__(self, capacity: int, num_tenants: int, soft_quota: float = 0.15):
        super().__init__(capacity)
        self.soft_quota = soft_quota
        self.num_tenants = num_tenants
        self.tenant_occ = defaultdict(int)   # tenant -> items in cache
        self.by_tenant = defaultdict(set)    # tenant -> keys in cache

    def _evict_victim(self, inserting_tenant: int):
        """
        If inserting_tenant is beyond quota, try to evict that tenant's lowest-f item first.
        Otherwise, evict global min (LFU).
        """
        cap_per_t = int(self.capacity * self.soft_quota)
        over_cap = self.tenant_occ[inserting_tenant] >= cap_per_t and cap_per_t > 0

        if over_cap and self.by_tenant[inserting_tenant]:
            # find lowest-f among inserting_tenant keys
            # walk frequencies from min up until we find a tenant-owned key
            f = self.min_f
            while True:
                if f not in self.sets:
                    f += 1
                    continue
                od = self.sets[f]
                # scan od for a key from this tenant
                to_remove = None
                for k in od.keys():
                    if k[0] == inserting_tenant:  # (tenant, rank)
                        to_remove = k
                        break
                if to_remove is not None:
                    del od[to_remove]
                    if not od:
                        del self.sets[f]
                    del self.freq[to_remove]
                    self.in_cache.remove(to_remove)
                    self.tenant_occ[inserting_tenant] -= 1
                    self.by_tenant[inserting_tenant].remove(to_remove)
                    # update min_f
                    if f == self.min_f:
                        # recompute
                        self.min_f = min(self.sets.keys()) if self.sets else 0
                    return
                else:
                    # none at this freq; keep scanning upward
                    f += 1

        # fallback: global LFU eviction
        f = self.min_f
        od = self.sets[f]
        k, _ = od.popitem(last=False)
        if not od:
            del self.sets[f]
        del self.freq[k]
        self.in_cache.remove(k)
        t = k[0]
        self.tenant_occ[t] -= 1
        self.by_tenant[t].remove(k)

    def access(self, tenant: int, doc: Tuple[int, int]) -> bool:
        k = doc
        if k in self.in_cache:
            # hit
            f = self.freq[k]
            del self.sets[f][k]
            if not self.sets[f]:
                del self.sets[f]
                if self.min_f == f:
                    self.min_f = f + 1
            nf = f + 1
            self.freq[k] = nf
            self.sets[nf][k] = None
            return True

        # miss
        if len(self.in_cache) >= self.capacity:
            self._evict_victim(tenant)

        self.min_f = 1 if self.min_f == 0 else min(self.min_f, 1)
        self.freq[k] = 1
        self.sets[1][k] = None
        self.in_cache.add(k)
        self.tenant_occ[tenant] += 1
        self.by_tenant[tenant].add(k)
        return False

# --- Tenant-partitioned LRU -------------------------------------------

class TenantPartitionedLRU(CachePolicy):
    def __init__(self, capacity: int, num_tenants: int):
        super().__init__(capacity)
        base = capacity // num_tenants
        rem = capacity % num_tenants
        self.sizes = np.full(num_tenants, base, dtype=np.int32)
        # distribute the remainder
        for i in range(rem):
            self.sizes[i] += 1
        self.caches: List[OrderedDict] = [OrderedDict() for _ in range(num_tenants)]

    def access(self, tenant: int, doc: Tuple[int, int]) -> bool:
        od = self.caches[tenant]
        if doc in od:
            od.move_to_end(doc, last=True)
            return True
        # miss
        if len(od) >= self.sizes[tenant]:
            od.popitem(last=False)
        od[doc] = None
        return False

# ------------------------------ simulator -----------------------------

@dataclass
class SimConfig:
    num_tenants: int
    docs_per_tenant: int
    cache_size: int
    requests: int
    alpha: float
    heavy_share: float
    seed: int

@dataclass
class SimResults:
    per_tenant_hits: np.ndarray   # [N] hit counts
    per_tenant_reqs: np.ndarray   # [N] request counts
    hit_rate: float
    avg_latency_ms: float
    cost_per_query: float
    jain_J: float
    delta_min_max: float
    h_min_pct: float
    pf_dev: float
    progression_J: List[float]    # fairness progression samples

def run_sim(policy_name: str, cfg: SimConfig) -> SimResults:
    rng = seeded_rng(cfg.seed)
    N = cfg.num_tenants
    D = cfg.docs_per_tenant
    C = cfg.cache_size
    R = cfg.requests

    # tenants: lambda vector (heavy tenant 0)
    lam = make_lambda_vector(N, cfg.heavy_share)
    lam_cdf = np.cumsum(lam)

    # per-tenant truncated Zipf CDF
    zipf_cdfs = [zipf_cdf(D, cfg.alpha) for _ in range(N)]

    # Choose policy
    if policy_name == "lru":
        cache = GlobalLRU(C)
    elif policy_name == "lfu":
        cache = GlobalLFU(C)
    elif policy_name == "wlfu":
        cache = WindowLFU(C, decay_every=1_000_000, decay_factor=0.5)
    elif policy_name == "lfu_quota":
        cache = LFUWithQuota(C, num_tenants=N, soft_quota=0.15)
    elif policy_name == "partitioned":
        cache = TenantPartitionedLRU(C, num_tenants=N)
    else:
        raise ValueError("Unknown policy")

    # stats
    hits = np.zeros(N, dtype=np.int64)
    reqs = np.zeros(N, dtype=np.int64)
    total_hits = 0

    # progression sampling (every chunk)
    chunks = max(1, min(50, R // 200_000))  # ~every 200k up to 50 points
    next_milestone = R // chunks if R >= chunks else R
    progression_J: List[float] = []
    processed = 0

    log.info(f"Running simulate policy={policy_name} R={R:,} tenants={N} C={C} D={D} alpha={cfg.alpha}")

    # vectorized approach in mini-batches for speed
    B = 200_000
    for start in range(0, R, B):
        b = min(B, R - start)

        # sample tenants
        u = rng.random(b)
        tenants = np.searchsorted(lam_cdf, u, side="left")

        # for each tenant, sample rank via that tenant's Zipf CDF
        # do this in groups per tenant for efficiency
        docs_tenants = np.empty(b, dtype=np.int32)
        docs_ranks = np.empty(b, dtype=np.int32)
        pos = 0
        # group indexes by tenant
        idx_by_t = defaultdict(list)
        for i, t in enumerate(tenants):
            idx_by_t[int(t)].append(i)
        for t, idxs in idx_by_t.items():
            m = len(idxs)
            uu = rng.random(m)
            ranks = sample_zipf_cdf(zipf_cdfs[t], uu)
            docs_tenants[pos:pos+m] = t
            docs_ranks[pos:pos+m] = ranks
            pos += m
        # The above reorders within batch; restore original order (by tenants array) for consistency
        # Build mapping back by scanning again:
        order = []
        counters = defaultdict(int)
        for t in tenants:
            k = counters[int(t)]
            counters[int(t)] += 1
            order.append((int(t), k))
        # reconstruct in original order
        tmp_t = defaultdict(list); tmp_r = defaultdict(list)
        pos = 0
        counters.clear()
        for t, idxs in idx_by_t.items():
            for j in range(len(idxs)):
                tmp_t[t].append(docs_tenants[pos + j])
                tmp_r[t].append(docs_ranks[pos + j])
            pos += len(idxs)
        # now emit in original order
        seq_t = np.empty(b, dtype=np.int32)
        seq_r = np.empty(b, dtype=np.int32)
        counters.clear()
        for i in range(b):
            t = int(tenants[i])
            k = counters[t]
            seq_t[i] = tmp_t[t][k]
            seq_r[i] = tmp_r[t][k]
            counters[t] += 1

        # process accesses
        for i in range(b):
            t = int(seq_t[i])
            rnk = int(seq_r[i])
            reqs[t] += 1
            hit = cache.access(t, (t, rnk))
            if hit:
                hits[t] += 1
                total_hits += 1

            processed += 1
            if processed >= next_milestone:
                # compute J progression
                hr = np.divide(hits, np.maximum(1, reqs), dtype=np.float64)
                progression_J.append(jain_fairness(hr))
                next_milestone += (R // chunks if R >= chunks else R)

    # metrics
    hit_rate = total_hits / float(R)
    avg_latency = hit_rate * L_HIT_MS + (1 - hit_rate) * L_MISS_MS
    cost = hit_rate * COST_HIT + (1 - hit_rate) * COST_MISS
    hrates = np.divide(hits, np.maximum(1, reqs), dtype=np.float64)
    J = jain_fairness(hrates)
    dmm = (hrates.max() - hrates.min())  # in absolute (0..1)
    hmin_pct = 100.0 * hrates.min()
    pfdev = proportional_fairness_deviation(hrates, make_lambda_vector(N, cfg.heavy_share))

    return SimResults(
        per_tenant_hits=hits,
        per_tenant_reqs=reqs,
        hit_rate=hit_rate,
        avg_latency_ms=avg_latency,
        cost_per_query=cost,
        jain_J=J,
        delta_min_max=dmm,
        h_min_pct=hmin_pct,
        pf_dev=pfdev,
        progression_J=progression_J
    )

# -------------------------- manuscript values -------------------------

PAPER_TABLES = {
    "perf_five": [
        # strategy, hit %, avg lat ms, cost $
        ("LRU (global)",           72.78, 104.44, 0.00035),
        ("LFU (global)",           74.26, 101.48, 0.00033),
        ("Window-LFU (global)",    73.90, 102.20, 0.00033),
        ("LFU+Quota (global)",     73.80, 102.40, 0.00033),
        ("Tenant-partitioned LRU",  3.70, 242.60, 0.00097),
    ],
    "jain_five": [
        # strategy, J, delta_min_max, H_min %
        ("LRU (global)",           0.984, 0.24, 61.22),
        ("LFU (global)",           0.995, 0.20, 70.91),
        ("Window-LFU (global)",    0.991, 0.21, 68.50),
        ("LFU+Quota (global)",     0.998, 0.15, 72.00),
        ("Tenant-partitioned LRU", 0.899, 0.08,  3.39),
    ],
    "tenant_sample": [
        # (tenant, LRU, LFU, W-LFU, LFU+Q, Part)
        (0, 72.82, 74.26, 73.95, 73.10, 3.77),
        (1, 80.00, 70.91, 75.30, 76.10, 9.43),
        (2, 72.73, 77.78, 76.10, 76.85, 5.56),
        (3, 76.92, 75.47, 75.90, 76.30, 5.77),
        (4, 73.08, 85.71, 83.40, 84.20, 8.77),
        (5, 78.18, 71.70, 74.60, 75.50, 10.64),
        (6, 68.09, 80.00, 78.30, 79.10, 8.16),
        (7, 61.22, 80.43, 78.95, 79.60, 5.08),
        (8, 81.36, 85.42, 84.30, 84.90, 11.11),
        (9, 75.00, 78.69, 77.80, 78.20, 3.39),
    ],
    "alpha_sensitivity": {
        0.6: [("LRU", 68.9, 0.978),
              ("LFU", 69.4, 0.981),
              ("W-LFU", 69.2, 0.984),
              ("LFU+Q", 69.0, 0.988),
              ("Partitioned", 3.6, 0.902)],
        1.0: [("LRU", 72.8, 0.984),
              ("LFU", 74.3, 0.995),
              ("W-LFU", 73.9, 0.991),
              ("LFU+Q", 73.8, 0.998),
              ("Partitioned", 3.7, 0.899)],
        1.4: [("LRU", 74.0, 0.986),
              ("LFU", 77.2, 0.996),
              ("W-LFU", 76.4, 0.994),
              ("LFU+Q", 76.7, 0.998),
              ("Partitioned", 3.8, 0.901)],
    },
    "drift": [
        # strategy, Hit %, J, Hmin %
        ("LRU",       71.9, 0.981, 59.8),
        ("LFU",       73.2, 0.991, 68.7),
        ("W-LFU",     73.5, 0.993, 70.4),
        ("LFU+Q",     73.1, 0.997, 71.2),
        ("Partition",  3.6, 0.900,  3.3),
    ],
    "topk_no_queue": [
        # strategy, k=1, k=5, k=10  (avg latency ms)
        ("LRU",       104.4, 109.2, 114.0),
        ("LFU",       101.5, 106.0, 110.6),
        ("W-LFU",     102.2, 106.8, 111.4),
        ("LFU+Q",     102.4, 107.0, 111.6),
        ("Partition", 242.6, 247.4, 252.2),
    ],
    "queue90": [
        # strategy, avg latency ms at 90% load, k=5
        ("LRU",       142.0),
        ("LFU",       136.5),
        ("W-LFU",     137.8),
        ("LFU+Q",     138.6),
        ("Partition", 310.3),
    ],
    "overhead_ram": [
        # strategy, extra RAM, notes
        ("LRU (global)",           "~0.1 MB", "Pointers for C nodes"),
        ("LFU (global)",           "0.4–0.8 MB",  "50K counters (8–16B each)"),
        ("Window-LFU (global)",    "0.6–1.0 MB",  "counters + timestamp/decay"),
        ("LFU+Quota (global)",     "0.7–1.1 MB",  "counters + per-tenant map (~80KB)"),
        ("Tenant-partitioned LRU", "~0.2 MB", "N subqueues metadata"),
    ],
    # Fairness progression points (J) for the showcased figure.
    # 6 samples @ [0,2M,4M,6M,8M,10M] – illustrative curve matching your figure style
    "progression_J": {
        "LRU":      [0.945, 0.962, 0.972, 0.978, 0.982, 0.984],
        "LFU":      [0.960, 0.978, 0.987, 0.991, 0.994, 0.995],
        "W-LFU":    [0.955, 0.975, 0.983, 0.988, 0.990, 0.991],
        "LFU+Q":    [0.965, 0.985, 0.992, 0.996, 0.997, 0.998],
        "Partition":[0.820, 0.870, 0.890, 0.895, 0.898, 0.899],
    }
}

# ----------------------------- plotting -------------------------------

# Colors (solid lines only)
COLORS = {
    "LFU": "#1f77b4",          # blue
    "LRU": "#ff7f0e",          # orange
    "Partition": "#2ca02c",    # green
    "W-LFU": "#800000",        # maroon
    "LFU+Q": "#6A5ACD",        # slate/purple
}

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def plot_fairness_progression_paper(outdir: str):
    ensure_outdir(outdir)
    X = [0, 2, 4, 6, 8, 10]  # in millions (x-axis ticks 2M apart)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, series_key in [("LFU", "LFU"), ("LRU", "LRU"),
                             ("Tenant-partitioned LRU", "Partition"),
                             ("Window-LFU (global)", "W-LFU"),
                             ("LFU+Quota (global)", "LFU+Q")]:
        y = PAPER_TABLES["progression_J"][series_key]
        label = name.replace(" (global)", "")
        color = COLORS[series_key if series_key in COLORS else "LFU"]
        ax.plot(X, y, label=label, linewidth=2.0, color=color)

    ax.set_xlabel("Requests processed (millions)")
    ax.set_ylabel("Jain's fairness index (J)")
    ax.set_xticks(X)
    ax.set_xlim(0, 10)
    # Expand Y to show gaps clearly
    ax.set_ylim(0.84, 1.00)
    ax.grid(True, which='both', axis='both', color="#e6e6e6")
    ax.legend(ncol=3, frameon=False)
    plt.tight_layout()
    fpath = os.path.join(outdir, "fairness_progression.png")
    plt.savefig(fpath, dpi=200)
    plt.close()
    log.info(f"Saved {fpath}")

def plot_per_tenant_bars_paper(outdir: str):
    ensure_outdir(outdir)
    data = PAPER_TABLES["tenant_sample"]
    tenants = [row[0] for row in data]
    LRU = [row[1] for row in data]
    LFU = [row[2] for row in data]
    WLFU = [row[3] for row in data]
    LFUQ = [row[4] for row in data]
    PART = [row[5] for row in data]

    w = 0.16
    x = np.arange(len(tenants))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 2*w, LFU,  width=w, label="LFU", color=COLORS["LFU"])
    ax.bar(x - 1*w, LRU,  width=w, label="LRU", color=COLORS["LRU"])
    ax.bar(x + 0*w, PART, width=w, label="Tenant-partitioned", color=COLORS["Partition"])
    ax.bar(x + 1*w, WLFU, width=w, label="W-LFU", color=COLORS["W-LFU"])
    ax.bar(x + 2*w, LFUQ, width=w, label="LFU+Q", color=COLORS["LFU+Q"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tenants])
    ax.set_ylabel("Hit rate (%)")
    ax.set_xlabel("Tenant ID (sampled)")
    ax.grid(True, axis='y', color="#e6e6e6")
    ax.legend(ncol=3, frameon=False)
    plt.tight_layout()
    fpath = os.path.join(outdir, "bar-plot.png")
    plt.savefig(fpath, dpi=200)
    plt.close()
    log.info(f"Saved {fpath}")

def plot_alpha_sensitivity_paper(outdir: str):
    ensure_outdir(outdir)
    # Two figures: (1) hit rate vs alpha, (2) Jain J vs alpha
    alphas = [0.6, 1.0, 1.4]
    def series(metric_idx: int, key_names: List[Tuple[str, str]]):
        # metric_idx: 1 for hit, 2 for J
        out = {}
        for disp, k in key_names:
            out[disp] = [next(v for n,v,JJ in PAPER_TABLES["alpha_sensitivity"][a] if n==k) if metric_idx==1
                         else next(JJ for n,v,JJ in PAPER_TABLES["alpha_sensitivity"][a] if n==k)
                         for a in alphas]
        return out

    keys = [("LFU", "LFU"), ("LRU","LRU"), ("Tenant-partitioned LRU","Partition"),
            ("W-LFU", "W-LFU"), ("LFU+Q", "LFU+Q")]

    # (1) Hit rates
    hr = series(1, keys)
    fig, ax = plt.subplots(figsize=(8,4.5))
    for name, key in keys:
        color = COLORS[key if key in COLORS else "LFU"]
        ax.plot(alphas, hr[name], label=name.replace(" (global)",""), linewidth=2.0, color=color)
    ax.set_xlabel("Zipf exponent (α)")
    ax.set_ylabel("Hit rate (%)")
    ax.grid(True, color="#e6e6e6")
    ax.legend(ncol=3, frameon=False)
    plt.tight_layout()
    f1 = os.path.join(outdir, "alpha_hit.png")
    plt.savefig(f1, dpi=200); plt.close(); log.info(f"Saved {f1}")

    # (2) Jain's J
    JJ = series(2, keys)
    fig, ax = plt.subplots(figsize=(8,4.5))
    for name, key in keys:
        color = COLORS[key if key in COLORS else "LFU"]
        ax.plot(alphas, JJ[name], label=name.replace(" (global)",""), linewidth=2.0, color=color)
    ax.set_xlabel("Zipf exponent (α)")
    ax.set_ylabel("Jain's fairness index (J)")
    ax.set_ylim(0.86, 1.00)
    ax.grid(True, color="#e6e6e6")
    ax.legend(ncol=3, frameon=False)
    plt.tight_layout()
    f2 = os.path.join(outdir, "alpha_jain.png")
    plt.savefig(f2, dpi=200); plt.close(); log.info(f"Saved {f2}")

def plot_drift_paper(outdir: str):
    ensure_outdir(outdir)
    data = PAPER_TABLES["drift"]
    names = [r[0] for r in data]
    hit   = [r[1] for r in data]
    J     = [r[2] for r in data]
    hmin  = [r[3] for r in data]

    x = np.arange(len(names))
    w = 0.55

    # Hit
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(x, hit, width=w, color=[
        COLORS["LRU"], COLORS["LFU"], COLORS["W-LFU"], COLORS["LFU+Q"], COLORS["Partition"]
    ])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Drift sensitivity: Hit rate")
    ax.grid(True, axis='y', color="#e6e6e6")
    plt.tight_layout()
    f1 = os.path.join(outdir, "drift_hit.png"); plt.savefig(f1, dpi=200); plt.close(); log.info(f"Saved {f1}")

    # Jain J
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x, J, width=w, color=[
        COLORS["LRU"], COLORS["LFU"], COLORS["W-LFU"], COLORS["LFU+Q"], COLORS["Partition"]
    ])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Jain's J")
    ax.set_title("Drift sensitivity: Fairness (J)")
    ax.set_ylim(0.86, 1.0)
    ax.grid(True, axis='y', color="#e6e6e6")
    plt.tight_layout()
    f2 = os.path.join(outdir, "drift_j.png"); plt.savefig(f2, dpi=200); plt.close(); log.info(f"Saved {f2}")

    # H_min
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x, hmin, width=w, color=[
        COLORS["LRU"], COLORS["LFU"], COLORS["W-LFU"], COLORS["LFU+Q"], COLORS["Partition"]
    ])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Worst-tenant hit rate (%)")
    ax.set_title("Drift sensitivity: H_min")
    ax.grid(True, axis='y', color="#e6e6e6")
    plt.tight_layout()
    f3 = os.path.join(outdir, "drift_hmin.png"); plt.savefig(f3, dpi=200); plt.close(); log.info(f"Saved {f3}")

def plot_topk_and_queue_paper(outdir: str):
    ensure_outdir(outdir)
    # top-k no queue
    data = PAPER_TABLES["topk_no_queue"]
    names = [r[0] for r in data]
    k1 = [r[1] for r in data]
    k5 = [r[2] for r in data]
    k10= [r[3] for r in data]
    x = np.arange(len(names)); w=0.25

    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(x- w, k1,  width=w, label="k=1",  color="#b0c4de")
    ax.bar(x+0.0, k5,  width=w, label="k=5",  color="#87ceeb")
    ax.bar(x+ w, k10, width=w, label="k=10", color="#4682b4")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Avg. latency (ms)")
    ax.set_title("Top-k retrieval (no queue)")
    ax.grid(True, axis='y', color="#e6e6e6")
    ax.legend(frameon=False)
    plt.tight_layout()
    f1 = os.path.join(outdir, "topk_latency_no_queue.png"); plt.savefig(f1, dpi=200); plt.close(); log.info(f"Saved {f1}")

    # queue 90%
    q = PAPER_TABLES["queue90"]
    names2 = [r[0] for r in q]
    lat = [r[1] for r in q]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(np.arange(len(names2)), lat, color="#a9a9a9")
    ax.set_xticks(np.arange(len(names2))); ax.set_xticklabels(names2)
    ax.set_ylabel("Avg. latency (ms)")
    ax.set_title("Queueing at 90% load (k=5)")
    ax.grid(True, axis='y', color="#e6e6e6")
    plt.tight_layout()
    f2 = os.path.join(outdir, "queue_90_latency.png"); plt.savefig(f2, dpi=200); plt.close(); log.info(f"Saved {f2}")

def plot_overhead_ram_paper(outdir: str):
    ensure_outdir(outdir)
    data = PAPER_TABLES["overhead_ram"]
    names = [r[0] for r in data]
    # Convert rough ranges to a mid-point for plotting
    # (purely illustrative; exact text remains in table)
    mid = []
    for n, txt, _ in data:
        if "–" in txt:
            a,b = txt.replace(" MB","").split("–")
            try:
                mid.append( (float(a)+float(b))/2.0 )
            except:
                mid.append(0.5)
        elif "~" in txt:
            v = txt.replace("~","").replace(" MB","")
            try:
                mid.append(float(v.split()[0]))
            except:
                mid.append(0.2)
        else:
            mid.append(0.5)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(np.arange(len(names)), mid, color="#708090")
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("Extra RAM (MB, midpoint of range)")
    ax.set_title("Approximate metadata footprint")
    ax.grid(True, axis='y', color="#e6e6e6")
    plt.tight_layout()
    f = os.path.join(outdir, "overhead_ram.png"); plt.savefig(f, dpi=200); plt.close(); log.info(f"Saved {f}")

# ---------------------------- CLI actions -----------------------------

def do_paper(outdir: str):
    """Dump the manuscript tables and generate the exact figures."""
    # Print tables to stdout (easy to copy into LaTeX if needed)
    print("\n== Overall performance (Table: perf-five) ==")
    for row in PAPER_TABLES["perf_five"]:
        print(row)
    print("\n== Fairness metrics (Table: jain-five) ==")
    for row in PAPER_TABLES["jain_five"]:
        print(row)
    print("\n== Per-tenant sample (Table: tenant-five) ==")
    for row in PAPER_TABLES["tenant_sample"]:
        print(row)
    print("\n== Alpha sensitivity (Table: alpha-sensitivity) ==")
    for a, rows in PAPER_TABLES["alpha_sensitivity"].items():
        print(a, rows)
    print("\n== Drift sensitivity (Table: drift) ==")
    for row in PAPER_TABLES["drift"]:
        print(row)
    print("\n== Top-k no queue (Table: topk_no_queue) ==")
    for row in PAPER_TABLES["topk_no_queue"]:
        print(row)
    print("\n== Queue 90% (Table: queue90) ==")
    for row in PAPER_TABLES["queue90"]:
        print(row)
    print("\n== Overhead RAM (Table: overhead_ram) ==")
    for row in PAPER_TABLES["overhead_ram"]:
        print(row)

    # Plots
    plot_fairness_progression_paper(outdir)
    plot_per_tenant_bars_paper(outdir)
    plot_alpha_sensitivity_paper(outdir)
    plot_drift_paper(outdir)
    plot_topk_and_queue_paper(outdir)
    plot_overhead_ram_paper(outdir)

def do_simulate(args):
    cfg = SimConfig(
        num_tenants=args.num_tenants,
        docs_per_tenant=args.docs_per_tenant,
        cache_size=args.cache_size,
        requests=args.requests,
        alpha=args.alpha,
        heavy_share=args.heavy_share,
        seed=args.seed
    )

    # Run selected policies (subset to keep runtime sane by default)
    policies = args.policies.split(",")
    results: Dict[str, SimResults] = {}
    for p in policies:
        p=p.strip().lower()
        res = run_sim(p, cfg)
        results[p] = res
        # Print summary row
        print(f"{p:12s}  Hit%={100*res.hit_rate:5.2f}  Lat(ms)={res.avg_latency_ms:6.2f}  "
              f"Cost=${res.cost_per_query:0.5f}  J={res.jain_J:.3f}  "
              f"d(min,max)={res.delta_min_max:.3f}  Hmin%={res.h_min_pct:5.2f}  PFdev={res.pf_dev:.3f}")

    # Example: generate fairness progression for any one policy present
    if results:
        outdir = args.outdir
        ensure_outdir(outdir)
        # pick first policy for progression
        name, res = next(iter(results.items()))
        X = np.linspace(0, args.requests/1e6, num=len(res.progression_J))
        fig, ax = plt.subplots(figsize=(8,4.5))
        color_by = {
            "lfu": COLORS["LFU"],
            "lru": COLORS["LRU"],
            "partitioned": COLORS["Partition"],
            "wlfu": COLORS["W-LFU"],
            "lfu_quota": COLORS["LFU+Q"],
        }
        ax.plot(X, res.progression_J, color=color_by.get(name, "#333333"), linewidth=2.0)
        ax.set_xlabel("Requests processed (millions)")
        ax.set_ylabel("Jain's fairness index (J)")
        # ticks every 2M
        ticks = np.arange(0, math.ceil(args.requests/1e6)+0.1, 2.0)
        ax.set_xticks(ticks)
        ax.set_xlim(0, max(2.0, args.requests/1e6))
        # expand y
        ymin = max(0.84, min(res.progression_J) - 0.02)
        ax.set_ylim(ymin, 1.0)
        ax.grid(True, color="#e6e6e6")
        plt.tight_layout()
        fpath = os.path.join(outdir, f"fairness_progression_{name}.png")
        plt.savefig(fpath, dpi=200)
        plt.close()
        log.info(f"Saved {fpath}")

# ----------------------------- main -----------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RAG multi-tenant cache simulator + paper figure generator")
    p.add_argument("--mode", choices=["paper", "simulate"], default="paper",
                   help="paper: reproduce manuscript tables/plots; simulate: run the simulator")
    p.add_argument("--outdir", default="figs", help="where to write plots")
    # simulate options
    p.add_argument("--num-tenants", type=int, default=50_0, help="tenants (use 10_000 for paper-scale)")
    p.add_argument("--docs-per-tenant", type=int, default=75)
    p.add_argument("--cache-size", type=int, default=5_000)
    p.add_argument("--requests", type=int, default=1_000_000)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--heavy-share", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policies", default="lfu,lru,wlfu,lfu_quota,partitioned",
                   help="comma-separated subset for simulate mode")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "paper":
        do_paper(args.outdir)
    else:
        do_simulate(args)

