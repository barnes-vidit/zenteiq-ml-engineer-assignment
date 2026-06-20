# Task 2B — Experimentation Log: Qwen Scaled Dense Model

---

## Qwen 1.092B — CPU

### Run 1 — FAILED (OOM)
**Config**: `base_emb_dim=1536`, `base_mlp_dim=4608`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8` (Qwen 1.092B configuration), `max_target_length=512`, `batch=1`, `attention=dot_product`, `dtype=bfloat16`, `remat_policy=full`, `scan_layers=true`
**Result**: JAX/XLA compilation OOM crash (process killed by OS, exit code 137) (see [Run_1_OOM_1.092B.log](../Logs/Qwen%20Scaled/Qwen%20scaled%20CPU/Run_1_OOM_1.092B.log)).  
**Cause**: Compiling a 1.092B parameter model under JAX/XLA exceeds the available 12 GB CPU host RAM during graph compilation.
**Fix**: Scaled down the model parameters to 0.732B by reducing hidden/MLP dimensions.

---

### Run 2 — Completed [0.732B]
**Config**: `base_emb_dim=1152`, `base_mlp_dim=3456`, `base_num_decoder_layers=28`, `base_num_query_heads=18`, `base_num_kv_heads=9`, `max_target_length=512`, `batch=1`, `attention=dot_product`, `dtype=bfloat16`, `remat_policy=full`, `scan_layers=true`
**Result**: 50 steps completed, ~96.75s/step (steady-state), ~5.29 tok/s, `total_weights=512`, loss 242.036 → 14.844 (loss at step 13: 96.599, step 17: 79.935, step 49: 14.844).
**RAM**: Total memory size: 9.6 GB (Output size: 4.1 GB, Temp size: 5.5 GB, Argument size: 4.1 GB).
**Fix**: Proactively scaled the model parameters up slightly to 0.767B to utilize the remaining available RAM.

---

### Run 3 — Completed [0.767B]
**Config**: `base_emb_dim=1216`, `base_mlp_dim=3648`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8`, `max_target_length=512`, `batch=1`, `attention=dot_product`, `dtype=bfloat16`, `remat_policy=full`, `scan_layers=true`
**Result**: 50 steps completed, ~102.14s/step (steady-state), ~5.01 tok/s, `total_weights=512`, loss 223.897 → 12.790 (loss at step 13: 96.251, step 17: 79.243, step 49: 12.790).
**RAM**: Total memory size: 10.0 GB (Output size: 4.3 GB, Temp size: 5.7 GB, Argument size: 4.3 GB).
**Fix**: Attempted to scale model size slightly higher but encountered OOM (Run 4).

---

### Run 4 — FAILED (OOM)
**Config**: tried to increase dimension slightly but got OOM.
**Result**: JAX/XLA compilation OOM crash.
**Cause**: Model parameter compilation buffers exceeded the 12 GB host memory limit on CPU.
**Fix**: Settled on 0.767B as the maximum trainable model size on CPU under host RAM constraints.

---

## Qwen 1.092B — GPU (T4)

### Run 1 — Completed
**Config**: batch=1, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16` (`base_emb_dim=1536`, `base_mlp_dim=4608`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8`)
**Result**: 50 steps completed, ~1.39s/step (steady-state), ~369.26 tok/s, `total_weights=512`, loss 315.844 → 0.064 (loss at step 13: 91.473, step 17: 63.781, step 49: 0.064).
**VRAM**: 6.1 GB / 10.92 GB (55.86%)
**Fix**: Loss collapsed rapidly due to small batch size (1). Increased batch size to 2 to normalize loss and utilize hardware.

---

### Run 2 — Completed
**Config**: batch=2, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~2.67s/step (steady-state), ~383.28 tok/s, `total_weights=1024`, loss 327.618 → 32.537 (loss at step 13: 111.470, step 17: 95.272, step 49: 32.537).
**VRAM**: 6.1 GB / 10.92 GB (55.86%)
**Fix**: Loss generalized successfully (no collapse). Increased batch size to 4 to scale up.

---

### Run 3 — Completed
**Config**: batch=4, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~4.94s/step (steady-state), ~415.79 tok/s, `total_weights=2048`, loss 337.428 → 71.177 (loss at step 13: 125.563, step 17: 111.476, step 49: 71.177).
**VRAM**: 6.1 GB / 10.92 GB (55.86%)
**Fix**: Increased batch size to 6 (Run 4) to test hardware boundaries.

---

### Run 4 — FAILED (OOM)
**Config**: batch=6, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: JAX/CUDA Out of Memory before training steps.
**Cause**: Training activation memory footprint exceeded the physical GPU VRAM capacity at batch=6.
**Fix**: Decreased batch size to 5.

---

### Run 5 — Completed
**Config**: batch=5, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~6.12s/step (steady-state), ~420.75 tok/s, `total_weights=2560`, loss 334.128 → 81.043 (loss at step 13: 127.146, step 17: 114.274, step 49: 81.043).
**VRAM**: 6.1 GB / 10.92 GB (55.86%)
**Fix**: Peak batch size of 5 reached for default rematerialization configuration. Explored performance benefits of custom rematerialization policies (`remat_policy=minimal`) at batch=2 (Run 6).

---

### Run 6 — Completed
**Config**: batch=2, remat policy = minimal, `max_target_length=512`, `attention=dot_product`, `dtype=bfloat16`
**Result**: 50 steps completed, ~2.41s/step (steady-state), ~426.93 tok/s, `total_weights=1024`, loss 327.746 → 32.500 (loss at step 13: 111.400, step 17: 95.235, step 49: 32.500).
**VRAM**: 6.1 GB / 10.92 GB (55.86%)
**Fix**: Using `remat_policy=minimal` reduced step time (from 2.67s to 2.41s) and increased throughput to 426.93 tok/s (highest among GPU runs) for batch=2.

---

## Qwen 1.092B — TPU (v5e-1)

### Run 1 — Completed
**Config**: batch=2, `max_target_length=512`, `attention=autoselected`, `dtype=bfloat16` (`base_emb_dim=1536`, `base_mlp_dim=4608`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8`)
**Result**: 50 steps completed, ~0.086s/step (steady-state), ~11,847.7 tok/s, `total_weights=1024`, loss 316.594 → 35.068 (loss at step 13: 111.572, step 17: 96.104, step 49: 35.068).
**HBM**: 6.13 GB / 15.75 GB (38.92%)
**Fix**: Memory usage is low. Increased batch size to 4.

---

### Run 2 — Completed
**Config**: batch=4, `max_target_length=512`, `attention=autoselected`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.139s/step (steady-state), ~14,688.9 tok/s, `total_weights=2048`, loss 316.097 → 76.826 (loss at step 13: 119.582, step 17: 108.597, step 49: 76.826).
**HBM**: 6.13 GB / 15.75 GB (38.92%)
**Fix**: Increased batch size to 8.

---

### Run 3 — Completed
**Config**: batch=8, `max_target_length=512`, `attention=autoselected`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.252s/step (steady-state), ~16,276.1 tok/s, `total_weights=4096`, loss 318.108 → 94.667 (loss at step 13: 128.243, step 17: 117.622, step 49: 94.667).
**HBM**: 6.13 GB / 15.75 GB (38.92%)
**Fix**: Since batch=8 achieved a peak throughput of ~16.3k tok/s with comfortable memory usage (38.92% HBM), we increased the batch size to 10 in Run 4 to search for the memory limit and check if throughput scaling continues.

---

### Run 4 — Completed
**Config**: batch=10, `max_target_length=512`, `attention=autoselected`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.316s/step (steady-state), ~16,221.3 tok/s, `total_weights=5120`, loss 315.055 → 98.730 (loss at step 13: 128.936, step 17: 118.879, step 49: 98.730).
**HBM**: 6.13 GB / 15.75 GB (38.92%)
**Fix**: Throughput dropped at batch=10 due to JAX layout transformations or padding. We increased batch size to 12 in Run 5 to verify if this throughput drop was a padding anomaly or a persistent scaling trend.
*Note: Corrected from FAILED (OOM) to Completed as the run executed successfully.*

---

### Run 5 — Completed
**Config**: batch=12, `max_target_length=512`, `attention=autoselected`, `dtype=bfloat16`
**Result**: 50 steps completed, ~0.373s/step (steady-state), ~16,485.4 tok/s, `total_weights=6144`, loss 318.426 → 101.320 (loss at step 13: 130.075, step 17: 120.193, step 49: 101.320).
**HBM**: 6.13 GB / 15.75 GB (38.92%)
**Fix/Insight**: TPU throughput peaked at batch=8 (~16.3k tok/s) and remained relatively stable at batch=10 (~16.2k tok/s) and batch=12 (~16.5k tok/s), showing the throughput ceiling of this model on TPU_0.