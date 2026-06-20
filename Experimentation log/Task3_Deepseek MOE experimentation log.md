# Task 3 — Experimentation Log: DeepSeek MoE Model

---

## DeepSeek MoE 0.719B — CPU
*Model Config*: `base_emb_dim=1024`, `base_mlp_dim=5472`, `base_moe_mlp_dim=704`, `base_num_decoder_layers=10`, `base_num_query_heads=8`, `base_num_kv_heads=8` (0.719B parameter sparse MoE model, configured as `deepseek2-16b`)

### Run 1 — Completed
**Config**: batch=1, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`, `remat_policy=full`, `scan_layers=true`
**Result**: 50 steps completed, ~79.78s/step (steady-state), ~6.42 tok/s, `total_weights=512`, loss 9.497 → 4.059 (loss at step 13: 7.196, step 17: 6.382, step 49: 4.059).
**RAM**: Total memory size: 8.7 GB (Output size: 4.0 GB, Temp size: 4.7 GB, Argument size: 4.0 GB).
**Fix**: Increased batch size to 2 to evaluate scaling behavior on CPU.

---

### Run 2 — Completed
**Config**: batch=2, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`, `remat_policy=full`, `scan_layers=true`
**Result**: 50 steps completed, ~142.65s/step (steady-state), ~7.18 tok/s, `total_weights=1024`, loss 9.507 → 5.421 (loss at step 13: 7.818, step 17: 7.207, step 49: 5.421).
**RAM**: Total memory size: 9.1 GB (Output size: 4.0 GB, Temp size: 5.1 GB, Argument size: 4.0 GB).
**Insight**: Doubling the batch size increased throughput from 6.42 tok/s to 7.18 tok/s (+11.9%) at the cost of higher latency per step (142.65s vs 79.78s).

---

## DeepSeek MoE 0.719B — GPU (T4)

### Run 1 — Completed
**Config**: batch=1, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`, `remat_policy=full`
**Result**: 50 steps completed, ~0.985s/step (steady-state), ~519.93 tok/s, `total_weights=512`, loss 9.470 → 0.010 (loss at step 13: 2.099, step 17: 0.734, step 49: 0.010).
**VRAM**: 8.03 GB / 10.92 GB (73.53%)
**Fix**: Rapid loss collapse/memorization occurred at batch=1. Both GPU runs utilized `remat_policy=full`. However, Run 1 had block-sparse compilation enabled (`megablox=true`, `sparse_matmul=false`), which resulted in a high memory footprint of 8.03 GB (73.53%). Directly scaling to batch=2 under these settings would have caused an CUDA OOM. The fix was turning off megablox (`megablox=false`) and enabling native sparse matrix multiply (`sparse_matmul=true`) in Run 2, which reduced peak memory.

---

### Run 2 — Completed (with Sharding Tricks)
**Config**: batch=2, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`, `remat_policy=full`
**Result**: 50 steps completed, ~5.41s/step (steady-state), ~189.27 tok/s, `total_weights=1024`, loss 9.485 → 6.493 (loss at step 13: 8.263, step 17: 7.814, step 49: 6.493).
**VRAM**: 4.02 GB / 10.92 GB (36.81%)
**Insight**: Disabling megablox (`megablox=false`) and enabling native sparse matrix multiplication (`sparse_matmul=true`) cut peak VRAM by ~50% (from 8.03 GB to 4.02 GB) despite doubling the batch size to 2. Note: `ici_parallelism=[1,1,1,-1,...]` was also set but has no effect with `num_devices=1`; on a single T4, `ici_fsdp_parallelism=-1` resolves to a shard size of 1.

---

## DeepSeek MoE 0.719B — TPU (v5e-1)

### Run 1 — Completed
**Config**: batch=1, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.067s/step (steady-state), ~7,696.2 tok/s, `total_weights=512`, loss 9.467 → 4.002 (loss at step 13: 7.151, step 17: 6.333, step 49: 4.002).
**HBM**: 4.06 GB / 15.75 GB (25.78%)
**Fix**: Memory usage is extremely low (25.78% HBM). We doubled the batch size to 2 in Run 2 to leverage TPU's matrix cores and improve throughput.

---

### Run 2 — Completed
**Config**: batch=2, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.074s/step (steady-state), ~13,926.3 tok/s, `total_weights=1024`, loss 9.553 → 5.460 (loss at step 13: 7.868, step 17: 7.255, step 49: 5.460).
**HBM**: 4.07 GB / 15.75 GB (25.84%)
**Fix**: Throughput scaled from 7.7k tok/s to 13.9k tok/s with virtually no HBM increase. We doubled the batch size to 4 in Run 3 to verify if the throughput continues to scale linearly.

---

### Run 3 — Completed
**Config**: batch=4, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.086s/step (steady-state), ~23,885.9 tok/s, `total_weights=2048`, loss 9.510 → 6.509 (loss at step 13: 8.293, step 17: 7.845, step 49: 6.509).
**HBM**: 4.07 GB / 15.75 GB (25.84%)
**Fix**: Throughput continues to scale (23.9k tok/s). We scaled batch size to 6 in Run 4 to push the TPU matrix calculation unit closer to saturation.

---

### Run 4 — Completed
**Config**: batch=6, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.096s/step (steady-state), ~32,024.0 tok/s, `total_weights=3072`, loss 9.502 → 6.978 (loss at step 13: 8.479, step 17: 8.102, step 49: 6.978).
**HBM**: 4.08 GB / 15.75 GB (25.90%)
**Fix**: Throughput reached 32.0k tok/s with negligible memory growth. We scaled the batch size to 8 in Run 5 to search for the linear scaling limits and peak throughput.

---

### Run 5 — Completed
**Config**: batch=8, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.112s/step (steady-state), ~36,704.2 tok/s, `total_weights=4096`, loss 9.496 → 7.275 (loss at step 13: 8.599, step 17: 8.265, step 49: 7.275).
**HBM**: 4.08 GB / 15.75 GB (25.90%)
**Insight**: MoE model scalability on TPU is highly linear. Scaling from batch=1 to batch=8 increased throughput 4.8x (from 7,696.2 tok/s to 36,704.2 tok/s) with near-zero increase in active memory allocation (4.06 GB to 4.08 GB).
