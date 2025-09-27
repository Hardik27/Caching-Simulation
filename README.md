# Caching at Scale: Efficiency & Fairness in Multi-Tenant RAG

This repository contains a compact Python simulator and figure generator for our paper:
**“Caching at Scale: Efficiency and Fairness Analysis in Multi-Tenant RAG Systems.”**
It provides executable code to reproduce all reported tables and plots, and includes a flexible simulator for five caching strategies.

---

## What’s Inside

- **sim.py:**
  The single, self-contained script for all simulation and figure generation.
    - **Paper mode:** Emits all tables and plots with the exact numbers reported in the manuscript.
    - **Simulation mode:** Runs a stochastic multi-tenant trace generator and applies the caching policies.

- **figs/**
  Created on demand; contains all output images used in the paper.

- **README.md:**  
  This file.

- **LICENSE:**
  (Optional) Add your preferred open-source license.

---

## Requirements

- **Python 3.9+**
- **numpy**
- **matplotlib**

Install Python dependencies:
pip install numpy matplotlib

---

## Quick Start

**Recreate all paper figures/tables (fast):**
python sim.py --mode paper --outdir figs

**Run a smaller, reproducible simulation (example):**
python sim.py --mode simulate --requests 1000000 --num-tenants 10000 --cache-size 50000 --alpha 1.0 --heavy-share 0.5 --policies lfu,lru,wlfu,lfu_quota,partitioned --outdir figs

---

## Simulator Modes

- **Paper Mode:**
  Uses frozen numbers, matching the manuscript (best for camera-ready figures and tables).
- **Simulation Mode:**
  Models multi-tenant dynamics; results may vary with different random seeds. Does not simulate request queueing or top-k latencies outside paper mode.

**Saving Fairness Progression:**
The fairness-progression curve is saved for the first policy in `--policies`. Run multiple times with different first policies to collect all progression curves.

---

## Caching Policies Implemented

- **LRU (global):** Single LRU across all tenants.
- **LFU (global):** O(1) LFU via frequency buckets.
- **Window-LFU (global):** LFU with decay/aging.
- **LFU+Quota (global):** LFU with soft per-tenant quota (default 15% occupancy).
- **Tenant-Partitioned LRU:** Equal per-tenant LRU slices (isolation baseline).

These map to:
- `LRUCache`
- `LFUCache`
- `WindowLFUCache`
- `LFUQuotaCache`
- `PartitionedLRUCache`
...within **sim.py**.

---

## Workload Model (Default)

- **Tenants:** N = 10,000 (tenant 0 is “heavy”)
- **Per-tenant documents:** D = 75 (ranks 1..75)
- **Requests:** Up to 10M (paper), examples use 1M for speed
- **Popularity:** Zipf(α), default α=1.0 (see α-sensitivity experiments)
- **Tenant mix:** λ0 = 0.5 for tenant 0; the rest distributed equally
- **Cache size:** C = 50,000 items

---

## Latency & Cost Model (Used for Tables)

- **Lhit:** 50 ms, **Lmiss:** 250 ms
- **Khit:** $0.0001 USD, **Kmiss:** $0.001 USD

---

## Fairness Metrics

With per-tenant hit rates \( H_i \) (fraction of tenant \( i \)'s requests hitting cache):

- **Jain’s index:**  
  \( J = \frac{(\sum_i H_i)^2}{N \cdot \sum_i H_i^2} \), range [0,1], higher is better.
- **Worst-tenant floor:**  
  \( H_{\text{min}} = \min_i H_i \) (reported as %)
- **Spread:**  
  \( \Delta = \max_i H_i - \min_i H_i \), lower is better.
- **Proportional Fairness Deviation (PF-Dev):**  
  \( \mathrm{PF{-}Dev} = \mathrm{mean}_i \frac{|H_i - \text{target}_i|}{\mathrm{mean}(H)} \)

PF-Dev printed in simulation logs (uniform tenant weights).

---

## Reproducing Paper Figures & Tables

python sim.py --mode paper --outdir figs

Creates:
- figs/fairness_progression.png
- figs/bar-plot.png
- figs/alpha_hit.png
- figs/alpha_jain.png
- figs/drift_hit.png
- figs/drift_j.png
- figs/drift_hmin.png
- figs/topk_latency_no_queue.png
- figs/queue_90_latency.png
- figs/overhead_ram.png

**Tables:** Printed to stdout (copy-paste ready).

---

## Simulation Examples

**Default scale, 1M requests:**
python sim.py --mode simulate --requests 1000000 --num-tenants 10000 --cache-size 50000 --alpha 1.0 --heavy-share 0.5 --policies lfu,lru,wlfu,lfu_quota,partitioned --outdir figs

**Different Zipf skew:**

python sim.py --mode simulate --requests 1000000 --alpha 0.6 --policies lfu,lru --outdir figs
python sim.py --mode simulate --requests 1000000 --alpha 1.4 --policies lfu,lru --outdir figs

**Save fairness progression for each policy (run one command per policy as first parameter):**

python sim.py --mode simulate --policies lfu,lru --outdir figs
python sim.py --mode simulate --policies lru,lfu --outdir figs
python sim.py --mode simulate --policies wlfu,lfu --outdir figs
python sim.py --mode simulate --policies lfu_quota,lfu --outdir figs
python sim.py --mode simulate --policies partitioned,lfu --outdir figs

**Console Output:**
Hit%, Avg Latency(ms), Cost($), J (Jain), Spread Δ(min,max), H_min%, PF-Dev.

---

## Reproducibility Checklist (Paper Base Case)

- Tenants N = 10,000; λ₀ = 0.5, others equal.
- Per-tenant docs D = 75; cache C = 50,000.
- Requests T = 10,000,000.
- Zipf α = 1.0 (unless otherwise stated).
- Latency/cost: Lhit = 50 ms, Lmiss = 250 ms; Khit = $0.0001, Kmiss = $0.001.
- Policies: LRU, LFU, Window-LFU, LFU+Quota (15%), Tenant-partitioned.
- Figures: paper mode uses canonical manuscript values.

---

## Limitations

- Simulation mode: Only single-document accesses (k=1); top-k and queueing results use paper-mode fixed numbers.
- Zipf-based traces; real deployments may vary.
- Partitioned LRU slices are equal; weighted/dynamic slices are future work.

---

## License

Add your preferred license file (e.g., MIT).

---

## Support

For issues or questions, open a GitHub issue or contact the authors listed in the paper.
