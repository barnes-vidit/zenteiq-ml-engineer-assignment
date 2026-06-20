# Experimentation Summary: Best-Performing Models Across Backends

This document summarizes the best-performing runs for each model-backend configuration, selected from the training logs based on:
1. **Tokens/sec/device** — The primary metric for throughput.
2. **TFLOPs/sec/device** — The primary metric for hardware compute efficiency (Model Flops Utilization).
3. **Step Time (seconds/step)** — The native step duration metric.

---

## 📊 Executive Summary Table

| Model Category | Backend | Best Run | Batch Size | Model Size (Params) | Architecture | Step Time (s) | Throughput (tok/s/dev) | Compute Efficiency (TFLOPs/s/dev) | Memory Utilization |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Qwen 0.6B** | **CPU** | Run 6 | 2 | 0.596B | `emb=1024`, `16q/8kv` | 145.687 | 7.029 | 0.026 | 9.1 GB Host RAM |
| **Qwen 0.6B** | **GPU (T4)** | Run 5 | 8 | 0.596B | `emb=1024`, `16q/8kv` | 5.503 | 744.255 | 2.792 | 30.49% VRAM (3.33 GB) |
| **Qwen 0.6B** | **TPU (v5e)** | Run 3 | 24 | 0.596B | `emb=1024`, `16q/8kv` | 0.464 | 26,492.122 | 99.400 | 21.33% HBM (3.36 GB) |
| **Qwen Scaled** | **CPU** ⚠️ | Run 2 | 1 | 0.732B | `emb=1152`, `18q/9kv` *(different shape)* | 96.179 | 5.323 | 0.024 | 9.6 GB Host RAM |
| **Qwen Scaled** | **GPU (T4)** | Run 6 | 2 | 1.092B | `emb=1536`, `16q/8kv` *(target arch)* | 2.405 | 425.709 | 2.865 | 55.86% VRAM (6.10 GB) |
| **Qwen Scaled** | **TPU (v5e)** | Run 5 | 12 | 1.092B | `emb=1536`, `16q/8kv` *(target arch)* | 0.373 | 16,485.418 | 110.932 | 38.92% HBM (6.13 GB) |
| **DeepSeek MoE**| **CPU** | Run 2 | 2 | 0.719B | `emb=1024`, MoE | 142.646 | 7.179 | 0.007 | 9.1 GB Host RAM |
| **DeepSeek MoE**| **GPU (T4)** ⚠️ | Run 2 | 2 | 0.719B | `emb=1024`, MoE | 5.410 | 189.274 | 0.173 | 36.81% VRAM (4.02 GB) |
| **DeepSeek MoE**| **TPU (v5e)** | Run 5 | 8 | 0.719B | `emb=1024`, MoE | 0.112 | 36,704.153 | 33.585 | 25.90% HBM (4.08 GB) |

Table values are taken from the final `completed step: 49` metric line of the selected raw training log. Best runs are selected by highest final logged throughput for each model/backend combination.

*⚠️ The Qwen Scaled CPU row uses a structurally different architecture (`emb_dim=1152`, `18q/9kv`) from the GPU/TPU target (`emb_dim=1536`, `16q/8kv`). It is a hardware capability bound measurement, not an apples-to-apples comparison. See the Qwen Scaled section below for details.*

*⚠️ The DeepSeek MoE GPU run is throughput-*regressed* relative to dense Qwen 0.6B on the same hardware/batch size (189.27 tok/s vs 641.62 tok/s) due to `remat_policy=full` recomputation overhead required to fit the model in T4 VRAM. This is the slowest of the three MoE backends relative to its dense counterpart — see the DeepSeek MoE section below for details.*

*Note: For the Qwen Scaled model on CPU, two configurations completed: the 0.732B model (highest throughput at 5.323 tok/s) and the 0.767B model (maximum parameter capacity trained on CPU at 5.01 tok/s). Both are detailed below. **These CPU configs use a different architecture shape than the GPU/TPU target** — see the ⚠️ note in the table above.*

---

## 🔍 Detailed Breakdown by Model

### 1. Qwen 0.6B Dense Model
*Model Config*: `base_emb_dim=1024`, `base_mlp_dim=3072`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8` (~596M parameters).

*   **CPU Backend (Best: Run 6)**
    *   *Log File*: [Run_6 qwen06b_cpu_512seq_2batch.log](../Logs/Qwen%200.6B/Qwen%200.6B%20CPU/Run_6%20qwen06b_cpu_512seq_2batch.log)
    *   *Throughput*: **7.029 tok/s**
    *   *TFLOPs/sec*: **0.026 TFLOPs/s**
    *   *Step Time*: **145.687 seconds/step**
    *   *Memory Footprint*: 9.1 GB Host RAM (Output: 3.3 GB, Temp: 5.8 GB, Argument: 3.3 GB).
    *   *Selection Rationale*: Outperformed Run 5 (batch=1, 6.60 tok/s) in raw token throughput (+5.8%), utilizing CPU cache/multi-threading layouts more efficiently despite the high per-step latency of 145.7s.

*   **GPU Backend (Best: Run 5)**
    *   *Log File*: [Run_5 qwen06b_gpu_8batch.log](../Logs/Qwen%200.6B/Qwen%200.6B%20GPU/Run_5%20qwen06b_gpu_8batch.log)
    *   *Throughput*: **744.255 tok/s**
    *   *TFLOPs/sec*: **2.792 TFLOPs/s**
    *   *Step Time*: **5.503 seconds/step**
    *   *Memory Footprint*: 9.6 GB compile-time RAM, utilizing **30.49% VRAM (3.33 GB)** on Nvidia Tesla T4.
    *   *Selection Rationale*: Batch size 8 represents the peak throughput achieved on the T4 GPU, maximizing parallelism without triggering OOM (VRAM remained stable at 3.33 GB due to weight-only static memory layouts).

*   **TPU Backend (Best: Run 3)**
    *   *Log File*: [Run_3 qwen06b_tpu_24batch.log](../Logs/Qwen%200.6B/Qwen%200.6B%20TPU/Run_3%20qwen06b_tpu_24batch.log)
    *   *Throughput*: **26,492.122 tok/s**
    *   *TFLOPs/sec*: **99.400 TFLOPs/s**
    *   *Step Time*: **0.464 seconds/step**
    *   *Memory Footprint*: 8.7 GB compile-time memory, utilizing **21.33% HBM (3.36 GB)** on TPU_0.
    *   *Selection Rationale*: Yielded the absolute peak throughput and highest compute utilization (99.40 TFLOPs, ~50.5% MFU) at batch size 24. Larger batch sizes (batch=32) degraded performance (dropping to ~25.1k tok/s) due to hardware/MXU saturation.

---

### 2. Qwen Scaled Dense Model

*   **CPU Backend — Deliberate Architecture Exploration (Best: Run 2 — Maximum Throughput)**
    > [!IMPORTANT]
    > The CPU backend **cannot run the 1.092B target architecture**. Run 1 attempted `emb_dim=1536` and OOM'd during XLA compilation before any training steps ran. Runs 2–3 each used an **independently-shaped architecture** chosen to fill the CPU's available compile-time RAM budget, while Run 4 failed OOM. These are not trimmed copies of the 1.092B model; they have different `emb_dim`, different MLP ratios, and different head counts. The CPU results characterise the **maximum parameter count the CPU backend can compile and run**, not cross-backend throughput of the same model.
    *   *Model Size*: **0.732 Billion parameters** (Config: `base_emb_dim=1152`, `base_mlp_dim=3456`, `base_num_decoder_layers=28`, `base_num_query_heads=18`, `base_num_kv_heads=9`)
    *   *Log File*: [Run_2_0.732B.txt](../Logs/Qwen%20Scaled/Qwen%20scaled%20CPU/Run_2_0.732B.txt)
    *   *Throughput*: **5.323 tok/s**
    *   *TFLOPs/sec*: **0.024 TFLOPs/s**
    *   *Step Time*: **96.179 seconds/step**
    *   *Memory Footprint*: 9.6 GB Host RAM (Output: 4.1 GB, Temp: 5.5 GB, Argument: 4.1 GB).
    *   *Selection Rationale*: Achieved the highest throughput on CPU (5.32 tok/s). Architecture shaped (`emb_dim=1152`, `18q/9kv`) to fit within the compile-time RAM budget after the 1.092B target OOM'd.

*   **CPU Backend (Run 3 — Maximum Parameter Capacity)**
    *   *Model Size*: **0.767 Billion parameters** (Config: `base_emb_dim=1216`, `base_mlp_dim=3648`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8`)
    *   *Log File*: [Run_3_0.767B.log](../Logs/Qwen%20Scaled/Qwen%20scaled%20CPU/Run_3_0.767B.log)
    *   *Throughput*: **4.982 tok/s**
    *   *TFLOPs/sec*: **0.024 TFLOPs/s**
    *   *Step Time*: **102.773 seconds/step**
    *   *Memory Footprint*: 10.0 GB Host RAM (Output: 4.3 GB, Temp: 5.7 GB, Argument: 4.3 GB).
    *   *Selection Rationale*: The absolute largest model configuration that successfully compiled and trained within the 12 GB RAM constraint before hitting OOM.

*   **GPU Backend (Best: Run 6)**
    *   *Model Size*: **1.092 Billion parameters** (Config: `base_emb_dim=1536`, `base_mlp_dim=4608`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8`)
    *   *Log File*: [Run6_1B_2batch_rematminimal.log](../Logs/Qwen%20Scaled/Qwen%20Scaled%20GPU/Run6_1B_2batch_rematminimal.log)
    *   *Throughput*: **425.709 tok/s**
    *   *TFLOPs/sec*: **2.865 TFLOPs/s**
    *   *Step Time*: **2.405 seconds/step**
    *   *Memory Footprint*: 10.8 GB compile-time RAM, utilizing **55.86% VRAM (6.10 GB)** on Nvidia Tesla T4.
    *   *Selection Rationale*: Utilizing the minimal rematerialization policy (`remat_policy=minimal`) at batch=2 reduced step time (from 2.41s to 2.41s) and boosted throughput to 425.71 tok/s, outperforming the larger batch size of 5 (420.75 tok/s).

*   **TPU Backend (Best: Run 5)**
    *   *Model Size*: **1.092 Billion parameters** (Config: `base_emb_dim=1536`, `base_mlp_dim=4608`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8`)
    *   *Log File*: [12batch.log](../Logs/Qwen%20Scaled/Qwen%20Scaled%20TPU/12batch.log)
    *   *Throughput*: **16,485.418 tok/s**
    *   *TFLOPs/sec*: **110.932 TFLOPs/s**
    *   *Step Time*: **0.373 seconds/step**
    *   *Memory Footprint*: 10.6 GB compile-time memory, utilizing **38.92% HBM (6.13 GB)** on TPU_0.
    *   *Selection Rationale*: Achieved the absolute peak throughput and efficiency (110.93 TFLOPs/device, ~56.3% MFU) at batch size 12.

---

### 3. DeepSeek MoE Model
*Model Config*: `base_emb_dim=1024`, `base_mlp_dim=5472`, `base_moe_mlp_dim=704`, `base_num_decoder_layers=10`, `base_num_query_heads=8`, `base_num_kv_heads=8` (~719M parameters, configured as `deepseek2-16b` sparse mixture-of-experts).

*   **CPU Backend (Best: Run 2)**
    *   *Log File*: [2batch.log](../Logs/Deepseek%20MOE/CPU/2batch.log)
    *   *Throughput*: **7.179 tok/s**
    *   *TFLOPs/sec*: **0.007 TFLOPs/s**
    *   *Step Time*: **142.646 seconds/step**
    *   *Memory Footprint*: 9.1 GB Host RAM (Output: 4.0 GB, Temp: 5.1 GB, Argument: 4.0 GB).
    *   *Selection Rationale*: Maximize CPU utilization and throughput, yielding +11.9% more tokens per second than batch=1 (6.42 tok/s). Slightly faster than dense Qwen 0.6B on CPU at the same batch size (7.18 vs 7.03 tok/s) — CPU is bottlenecked elsewhere (compile-time RAM, no hardware MXU), so the sparsity advantage barely registers.

*   **GPU Backend (Best: Run 2)**
    > [!IMPORTANT]
    > This is a throughput **regression**, not a win. MoE on GPU (189.27 tok/s) is ~3.4x **slower** than dense Qwen 0.6B at the same batch size (641.62 tok/s) — the opposite of the TPU result below. `remat_policy=full` was required to fit the model in the T4's 16 GB VRAM, and the recomputation overhead it introduces outweighs the sparse-routing compute savings on this hardware. See the Task 3 writeup for full reasoning.
    *   *Log File*: [2batch_withtricks.log](../Logs/Deepseek%20MOE/GPU/2batch_withtricks.log)
    *   *Throughput*: **189.274 tok/s**
    *   *TFLOPs/sec*: **0.173 TFLOPs/s**
    *   *Step Time*: **5.410 seconds/step**
    *   *Memory Footprint*: 8.5 GB compile-time memory, utilizing **36.81% VRAM (4.02 GB)** on Nvidia Tesla T4.
    *   *Selection Rationale*: Both GPU runs used `remat_policy=full`. Peak memory in Run 1 (batch=1) was 73.53% (8.03 GB) due to MegaBlox compilers (`megablox=true`, `sparse_matmul=false`). Turning off megablox (`megablox=false`) and enabling native sparse matmul (`sparse_matmul=true`) in Run 2 (batch=2) optimized memory and reduced VRAM to 36.81% (4.02 GB), freeing enough VRAM to double the batch size to 2, the highest batch size achieved for GPU MoE — but at the cost of step time, since rematerialization re-runs the forward pass during the backward pass. Net throughput (189.27 tok/s) is markedly below the dense Qwen 0.6B GPU baseline at the same batch size. The `ici_parallelism=[1,1,1,-1,...]` flag was present in the run but is a no-op on a single T4 (`num_devices=1`).

*   **TPU Backend (Best: Run 5)**
    *   *Log File*: [8batch.log](../Logs/Deepseek%20MOE/TPU/8batch.log)
    *   *Throughput*: **36,704.153 tok/s**
    *   *TFLOPs/sec*: **33.585 TFLOPs/s**
    *   *Step Time*: **0.112 seconds/step**
    *   *Memory Footprint*: 7.6 GB compile-time memory, utilizing **25.90% HBM (4.08 GB)** on TPU_0.
    *   *Selection Rationale*: Represents the absolute peak throughput achieved for the MoE model across all devices. The sparse execution paths of MoE scaled highly linearly, yielding 4.8x higher throughput at batch=8 compared to batch=1 (7,696 tok/s) with virtually no extra HBM memory growth. Unlike the GPU run, no rematerialization was needed on TPU at this batch size, so MoE's sparsity advantage shows through cleanly here (36,704 tok/s vs 26,492 tok/s for dense Qwen 0.6B, ~38.6% faster).