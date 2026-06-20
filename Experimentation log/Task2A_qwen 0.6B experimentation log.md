# Task 2A — Experimentation Log: Qwen 0.6B Dense Model

---

## Qwen 0.6B — CPU

### Run 1 — FAILED (Pallas kernel crash)
**Config**: default flags, no `attention` specified  
**Result**: `ValueError: Only interpret mode is supported on CPU backend`  
**Cause**: MaxText defaults to Flash Attention (Pallas kernel), which doesn't compile on CPU  
**Fix**: added `attention=dot_product`

---

### Run 2 — FAILED (OOM, exit code 137)
**Config**: `attention=dot_product`, `per_device_batch_size=1`, `max_target_length=128`  
**Result**: killed by OS after `Output size: 6.7 GB` on a 12 GB RAM machine  
**Cause**: XLA compilation buffers for 0.6B at seq=128 exceed available RAM  
**Fix**: added `remat_policy=full` + `scan_layers=true` + dropped `max_target_length=32`

---

### Run 3 — Completed, REJECTED
**Config**: `max_target_length=32`, `batch=1`, `remat_policy=full`, `scan_layers=true`  
**Result**: 50 steps completed, ~25s/step, 1.28 tok/s, `total_weights=32`  
**Problem**: loss collapsed to 0.000 at **step 13** — model memorised 32-token batch in 13 steps. 37 steps wasted.  
**Fix**: increased `max_target_length=64`

---

### Run 4 — Completed, REJECTED
**Config**: `max_target_length=64`, everything else same  
**Result**: 50 steps, ~25s/step, 2.5 tok/s, `total_weights=64`  
**Problem**: loss collapsed at **step 17** — same issue, just 4 steps later. Root cause unchanged: 596M params >> 64 tokens/step.  
**Insight**: CPU memory ceiling makes real training signal impossible at small seq. Abandoned seq-tweaking approach.  
**Fix**: switched to `max_target_length=512` with `remat_policy=full` + `scan_layers=true` to stay within RAM

---

### Run 5 — Completed, REJECTED
**Config**: `max_target_length=512`, `batch=1`, `remat_policy=full`, `scan_layers=true`, `attention=dot_product`  
**Result**: 50 steps, ~77s/step, ~6.7 tok/s, `total_weights=512`, loss 242.638 → 20.934, no collapse  
**RAM**: Total 8.0 GB, Output size: 3.3 GB, Temp size: 4.7 GB, Argument size: 3.3 GB  
**Fix**: Since training succeeded without memory issue, we increased the batch size to 2 (Run 6) to test CPU throughput scaling.

---

### Run 6 — Completed
**Config**: `max_target_length=512`, `batch=2`, `remat_policy=full`, `scan_layers=true`, `attention=dot_product`  
**Result**: 50 steps, ~147s/step, ~7.0 tok/s, `total_weights=1024`, loss 237.394 → 54.566, no collapse  
**RAM**: Memstats unavailable on CPU (XLA CPU backend does not report per-op memory). Compile-time memory: Total 9.1 GB, Output size: 3.3 GB, Temp size: 5.8 GB, Argument size: 3.3 GB

---

## Qwen 0.6B — GPU (T4)

### Run 1 — FAILED (JAX plugin version conflict)
**Config**: first attempt, no `JAX_PLATFORMS` set  
**Result**: `AttributeError: register_custom_type_id_handler` — Colab CUDA plugin (jaxlib 0.7.2) conflicted with MaxText's jaxlib (0.10.0)  
**Fix**: `pip uninstall jax-cuda12-plugin jax-cuda12-pjrt` + `pip install "jax[cuda12]" --upgrade`; added `JAX_PLATFORMS=gpu`

---

### Run 2 — Completed
**Config**: `max_target_length=512`, `batch=2`, `attention=dot_product`, `dtype=bfloat16`  
**Result**: 50 steps, ~1.596s/step (steady-state), ~641 tok/s, `total_weights=1024`, loss 237.047 → 54.250, no collapse  
**VRAM**: 3.33 GB / 10.92 GB (30.49%)  
**Fix**: Memory usage is low (30.49% VRAM). Doubled batch size to 4 in Run 3 to test scaling capacity and maximize GPU utilization.

---

### Run 3 — Completed
**Config**: `max_target_length=512`, `batch=4`, `attention=dot_product`, `dtype=bfloat16`  
**Result**: 50 steps, ~2.874s/step (steady-state), ~712 tok/s, `total_weights=2048`, loss 235.646 → 76.836, no collapse  
**VRAM**: 3.33 GB / 10.92 GB (30.49%)  
**Fix**: Throughput scaled to 712 tok/s while VRAM stayed flat. Increased batch size to 6 in Run 4 to push the GPU hardware limits.

---

### Run 4 — Completed (50 steps)
**Config**: `max_target_length=512`, `batch=6`, `attention=dot_product`, `dtype=bfloat16`  
**Result**: 50 steps, ~4.287s/step (steady-state), ~716 tok/s, `total_weights=3072`, loss 234.398 → 86.067, no collapse  
**VRAM**: 3.33 GB / 10.92 GB (30.49%)  
**Note**: Loss higher than Run 3 at step 49 — larger batch means fewer gradient updates relative to data volume in 50 steps  
**Fix**: Throughput gains are saturating but VRAM is still flat. Increased batch size to 8 in Run 5 to identify the peak throughput boundary.

---

### Run 5 — Completed
**Config**: `max_target_length=512`, `batch=8`, `attention=dot_product`, `dtype=bfloat16`  
**Result**: 50 steps, ~5.503s/step (steady-state), ~744 tok/s, `total_weights=4096`, loss 233.904 → 91.154, no collapse  
**VRAM**: 3.33 GB / 10.92 GB (30.49%)  
**Note**: Throughput (tok/s) is highest among GPU runs due to better batch parallelism; loss at step 49 is higher because 50 steps covers fewer "passes" over the data at batch=8

---

## Qwen 0.6B — TPU (v5e-1)

### Run 1 — Completed
**Config**: `max_target_length=512`, `batch=4`, `attention=autoselected`, `dtype=bfloat16`  
**Result**: 50 steps, ~0.090s/step (steady-state), ~22,682 tok/s, `total_weights=2048`, loss 235.908 → 79.117  
**HBM**: 3.36 GB / 15.75 GB (21.33%)  
**Fix**: Memory usage is low (21.33% HBM). Scaled batch size significantly to 16 in Run 2 to test scaling behavior.

---

### Run 2 — Completed
**Config**: `max_target_length=512`, `batch=16`, `attention=autoselected`, `dtype=bfloat16`  
**Result**: 50 steps, ~0.314s/step (steady-state), ~26,052 tok/s, `total_weights=8192`, loss 234.101 → 97.038  
**HBM**: 3.36 GB / 15.75 GB (21.33%)  
**Fix**: Throughput scaled to 26k tok/s. Increased batch size to 24 in Run 3 to test scaling limits and look for throughput saturation.

---

### Run 3 — Completed
**Config**: `max_target_length=512`, `batch=24`, `attention=autoselected`, `dtype=bfloat16`  
**Result**: 50 steps, ~0.464s/step (steady-state), ~26,492 tok/s, `total_weights=12288`, loss 233.398 → 98.547  
**HBM**: 3.36 GB / 15.75 GB (21.33%)  
**Fix**: Throughput scaling is plateauing. Increased batch size to 32 in Run 4 to find the peak throughput ceiling of the hardware.

---

### Run 4 — Completed
**Config**: `max_target_length=512`, `batch=32`, `attention=autoselected`, `dtype=bfloat16`  
**Result**: 50 steps, ~0.651s/step (steady-state), ~25,162 tok/s, `total_weights=16384`, loss 234.050 → 100.943  
**HBM**: 3.36 GB / 15.75 GB (21.33%)
