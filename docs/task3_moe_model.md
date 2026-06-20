# DeepSeek MoE Model Benchmarking

> [!NOTE]
> **Scaling Strategy**: To scale down the Mixture-of-Experts architecture under the 1B parameter threshold (~719M parameters), I studied the **DeepSeek-V2-16B** model configuration (often referred to as DeepSeek-V2-Lite). The downscaling was implemented by applying an exact **halving (0.5x scaling)** to the primary dimensional and expert parameters of the DeepSeek-V2-16B base model: reducing `base_emb_dim` from 2048 to 1024, the expert MLP size (`base_moe_mlp_dim`) from 1408 to 704, the total routed experts from 64 to 32, active routed experts per token from 6 to 3, and shared experts from 2 to 1. To maintain model alignment, the dense MLP dimension was kept at DeepSeek-V2-16B's standard value (`base_mlp_dim=5472`), while decoder layers were reduced to 10.
*The pattern in one line: width-of-the-model knobs (emb_dim, heads, FFN, experts, depth) scale; the MLA attention "recipe" (head dims + lora rank) and all the routing/RoPE/norm hyperparameters stay fixed. That's literally how DeepSeek went from 16B to 671B — same recipe, wider and deeper.*



## 📊 Performance Summary (Best Runs)

| Backend | Best Run | Model Size (Total) | Batch Size | Step Time | Throughput | MFU (TFLOPs/s/dev) | Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CPU** | Run 2 | 0.719B | 2 | 142.65s | 7.18 tok/s | 0.007 | 9.1 GB Host RAM |
| **GPU (T4)** | Run 2 | 0.719B | 2 | 5.41s | 189.27 tok/s | 0.173 | 4.02 GB VRAM |
| **TPU (v5e)** | Run 5 | 0.719B | 8 | 0.112s | 36,704.15 tok/s | 33.585 | 4.08 GB HBM |

> [!WARNING]
> **CPU Routing Caveat**: The [`deepseek_moe_cpu.yml`](../configs/deepseek_moe_cpu.yml) config does **not** explicitly set the MoE routing parameters (`num_experts`, `num_experts_per_tok`, `shared_experts`) or the MLA attention dimensions. The GPU and TPU configs override these to 32 experts / 3 active / 1 shared. If the CPU run inherited the `deepseek2-16b` model defaults instead (64 experts, 6 active), it would have run a **different routing configuration** than documented — and would not be true sparse MoE in the same sense as the GPU/TPU runs. The CPU throughput figure (7.18 tok/s) should therefore be treated with this uncertainty in mind.

### Throughput comparison across backends
```text
CPU (1x)       | 7.18 tok/s
               | ▏
GPU T4 (27x)   | 189.27 tok/s
               | ▎
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
* **MoE throughput vs dense**: The throughput advantage of DeepSeek MoE over dense models is highly dependent on the hardware backend:
  * **On TPU**: MoE achieves a dramatic speedup, reaching **36,704.15 tok/s** (batch=8) compared to **26,492.12 tok/s** for Qwen 0.6B (~38.5% faster) and **16,485.42 tok/s** for the larger Qwen Scaled (1.09B) model (~2.23x faster).
  * **On GPU**: MoE is significantly slower, achieving only **189.27 tok/s** (batch=2) compared to **744.26 tok/s** for Qwen 0.6B at batch size 8 (and **641.62 tok/s** at batch size 2). This performance regression is due to activation rematerialization overhead (`remat_policy=full`) required to fit MoE in the T4 GPU's VRAM, as well as sparse routing kernel overheads.
  * **On CPU**: MoE achieves **7.18 tok/s** (batch=2), which is slightly faster than Qwen 0.6B (**7.03 tok/s**). ⚠️ *See the routing caveat above — the CPU config does not explicitly set routing params, so it is unverified whether this run used true sparse MoE routing (32 experts / 3 active) or the `deepseek2-16b` defaults (64 experts / 6 active). Either way, CPU has no hardware sparse-matmul support, so the sparsity benefit would not materially affect throughput here.*
* **Lower TFLOPs/s/dev (MFU)**: The Model Flops Utilization metrics are much lower for the MoE runs (e.g., 33.585 TFLOPs/s/dev on TPU vs. 110.932 TFLOPs/s/dev for Qwen Scaled). This is because the MFU metric calculates FLOPs based on the *total* parameter count (assuming all experts run), whereas the hardware is only doing compute work for the active subset of parameters.
* **Activation Rematerialization & Compiler Optimizations**: On GPU, both runs utilized `remat_policy=full`. Run 1 (batch=1, using MegaBlox compiler flags `megablox=true`, `sparse_matmul=false`) consumed 73.53% VRAM (8.03 GB). Directly scaling to batch=2 under these settings would have OOM'd. Run 2 optimized memory by turning off MegaBlox (`megablox=false`) and enabling native sparse path (`sparse_matmul=true`), reducing VRAM to 36.81% (4.02 GB) and enabling batch size 2. The `ici_parallelism=[1,1,1,-1,...]` flag present in the run config has no effect on a single-device T4 (`ici_fsdp_parallelism=-1` resolves to a shard count of 1 with `num_devices=1`).
