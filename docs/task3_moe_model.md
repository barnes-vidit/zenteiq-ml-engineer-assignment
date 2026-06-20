# DeepSeek MoE Model Benchmarking

> [!NOTE]
> **Scaling Strategy**: To scale down the Mixture-of-Experts architecture under the 1B parameter threshold (~719M parameters), I studied the **DeepSeek-V2-16B** model configuration (often referred to as DeepSeek-V2-Lite). The downscaling was implemented by applying an exact **halving (0.5x scaling)** to the primary dimensional and expert parameters of the DeepSeek-V2-16B base model: reducing `base_emb_dim` from 2048 to 1024, the expert MLP size (`base_moe_mlp_dim`) from 1408 to 704, the total routed experts from 64 to 32, active routed experts per token from 6 to 3, and shared experts from 2 to 1. To maintain model alignment, the dense MLP dimension was kept at DeepSeek-V2-16B's standard value (`base_mlp_dim=5472`), while decoder layers were reduced to 10.



## 📊 Performance Summary (Best Runs)

| Backend | Best Run | Model Size (Total) | Batch Size | Step Time | Throughput | MFU (TFLOPs/s/dev) | Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CPU** | Run 2 | 0.719B | 2 | 144.36s | 7.10 tok/s | 0.007 | 9.1 GB Host RAM |
| **GPU (T4)** | Run 2 | 0.719B | 2 | 5.21s | 1015.49 tok/s | 0.929 | 4.02 GB VRAM |
| **TPU (v5e)** | Run 5 | 0.719B | 8 | 0.112s | 36,704.15 tok/s | 33.585 | 4.08 GB HBM |

### 📈 Throughput Comparison & Speedup Scale
```text
CPU (1x)       | 7.10 tok/s
               | ▏
GPU T4 (143x)  | 1015.49 tok/s
               | █
TPU v5e (5169x)| 36,704.15 tok/s
               | █████████████████████████████████████████
```

---

## 🔍 Key Interpretations

### 1. Scaling Under 1B Parameters
The default DeepSeek-V2 architecture is extremely large (16B+ parameters). To bring it under 1B total parameters, the configuration overrides scaled down the core dimensions:
* `base_emb_dim`: 1024
* `base_mlp_dim`: 5472
* `base_moe_mlp_dim`: 704
* `base_num_decoder_layers`: 10
* `base_num_kv_heads`: 8
* `base_num_query_heads`: 8

This configuration yields a total parameter count of **0.719 Billion**. 

### 2. Active vs. Total Parameters in MoE
* **Active Parameters**: Though the model has **719M total parameters**, only a small fraction is active per token during forward/backward passes.
* **Routing Logic**: Each layer contains 32 experts. The gating network dynamically routes each token to only 3 active experts (`num_experts_per_tok=3`), plus 1 shared expert that always runs (`shared_experts=1`). 
* This means only 4 out of 33 possible MLP pathways are evaluated per token, dramatically reducing the FLOPs required per token relative to the model's total size.

### 3. Comparison to Dense (Qwen) Runs
* **Higher Throughput**: DeepSeek MoE achieves significantly higher throughput than dense models of comparable sizes. For example:
  * **On GPU**: MoE achieves **1,015.49 tok/s** (batch=2) compared to **641.00 tok/s** for Qwen 0.6B at the same batch size.
  * **On TPU**: MoE reaches **36,704.15 tok/s** (batch=8) compared to **26,052.58 tok/s** for Qwen 0.6B and **16,276.12 tok/s** for the larger Qwen Scaled (1.09B) model.
* **Lower TFLOPs/s/dev (MFU)**: The Model Flops Utilization metrics are much lower for the MoE runs (e.g., 33.585 TFLOPs/s/dev on TPU vs. 109.524 TFLOPs/s/dev for Qwen Scaled). This is because the MFU metric calculates FLOPs based on the *total* parameter count (assuming all experts run), whereas the hardware is only doing compute work for the active subset of parameters.
* **Activation Rematerialization Impact**: On GPU, Run 1 (batch=1, no `remat_policy`) consumed 73.53% VRAM (8.03 GB) storing all intermediate activations. Scaling directly to batch=2 would have OOM'd. Run 2 enabled `remat_policy=full`, which recomputes forward-pass activations on demand during the backward pass instead of storing them in VRAM — halving peak activation memory and allowing batch=2 to comfortably fit within 4.02 GB (36.81% VRAM). The `ici_parallelism=[1,1,1,-1,...]` flag present in the run config has no effect on a single-device T4 (`ici_fsdp_parallelism=-1` resolves to a shard count of 1 with `num_devices=1`).
