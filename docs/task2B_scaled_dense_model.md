# Qwen Scaled Dense Model Benchmarking

> [!NOTE]
> **Scaling Strategy**: To scale the dense model from 0.6B to ~1.1B parameters, I studied the configuration ratios of the **Qwen3** model family. The target architecture is a custom interpolation between **Qwen3-0.6B** and **Qwen3-1.7B**: `base_emb_dim=1536`, `base_mlp_dim=4608` (3× emb_dim), 28 decoder layers, and a 16q/8kv GQA ratio (2:1). This resulted in a stable **1.092 Billion** parameter model, which ran on GPU and TPU.
>
> **CPU Architecture Note**: The CPU backend could not compile the 1.092B model (XLA graph compilation OOM at 12 GB host RAM). The CPU runs used a **structurally different, independently-shaped architecture** (`base_emb_dim=1152`, `base_mlp_dim=3456`, `18q/9kv` heads) — not a compressed version of the 1.092B model. This config was chosen to maximally exploit the CPU's available compile-time RAM budget. The CPU row in the table below is therefore a **hardware capability bound measurement**, not a direct apples-to-apples throughput comparison with GPU/TPU.



## 📊 Performance Summary (Best Runs)

| Backend | Best Run | Model Size | Architecture | Batch Size | Step Time | Throughput | MFU (TFLOPs/s/dev) | Memory |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **CPU** ⚠️ | Run 2 | 0.732B | `emb=1152`, `18q/9kv` (different shape) | 1 | 96.18s | 5.32 tok/s | 0.024 | 9.6 GB Host RAM |
| **GPU (T4)** | Run 6 | 1.092B | `emb=1536`, `16q/8kv` (target arch) | 2 | 2.41s | 425.71 tok/s | 2.865 | 6.10 GB VRAM |
| **TPU (v5e)** | Run 5 | 1.092B | `emb=1536`, `16q/8kv` (target arch) | 12 | 0.373s | 16,485.42 tok/s | 110.932 | 6.13 GB HBM |

*⚠️ The CPU run uses a different model shape than GPU/TPU. Cross-backend throughput comparison for the CPU row is not apples-to-apples — see the Scaling Strategy note above.*

### Throughput comparison across backends
```text
CPU (1x)       | 5.32 tok/s
               | ▏
GPU T4 (80x)   | 425.71 tok/s
               | █▌
TPU v5e (3100x)| 16,485.42 tok/s
               | ███████████████████████████
```

---

## 🔍 Key Interpretations

### 1. Backend Performance Variance
* **Hardware scaling gaps**: Similar to the base model, CPU throughput is extremely low due to DDR memory bandwidth limitations. TPU v5e delivers ~3100x speedup vs. CPU, processing 16,485.42 tok/s because the larger model runs natively on hardware systolic arrays.
* **TPU MFU by model size**: The compute efficiency (MFU) on TPU v5e scaled from **99.400 TFLOPs/s/dev** (on 0.6B, ~50.5% MFU) to **110.932 TFLOPs/s/dev** (on 1.092B, ~56.3% MFU). This is because the larger layer dimension (`base_emb_dim=1536`) aligns better with the TPU v5e Matrix Multiply Unit (MXU) tile sizes (multiples of 128/256), leading to higher hardware utilization and less compiler-introduced padding.

### 2. Model Scaling Strategy & Hardware Constraints
* **GPU & TPU (1.092B — Target Architecture)**: The intended scaled model uses `base_emb_dim=1536`, `base_mlp_dim=4608`, `base_num_decoder_layers=28`, `base_num_query_heads=16`, `base_num_kv_heads=8`. This is the architecture the assignment asked to scale toward.
  * On GPU T4, this model was made stable at batch size 2 by setting `remat_policy=minimal` (reducing step time to 2.41s at 6.10 GB VRAM).
* **CPU Backend — Deliberate Architecture Exploration, Not a Comparison Run**: The 1.092B target architecture (`emb_dim=1536`) was attempted on CPU first (Run 1) and immediately OOM'd during XLA graph compilation — the compiler's static buffers alone exceeded the available 12 GB host RAM before a single training step ran (see [Run_1_OOM_1.092B.log](../Logs/Qwen%20Scaled/Qwen%20scaled%20CPU/Run_1_OOM_1.092B.log)).
  * Rather than treating this as a failed benchmark, Runs 2–3 deliberately explored the **maximum parameter count achievable on CPU** as a hardware characterisation exercise, while Run 4 failed OOM. Each successful run used a freshly-shaped architecture chosen to fill the remaining RAM budget:
    * **Run 2 (0.732B)**: `emb_dim=1152`, `18q/9kv` heads — highest throughput at 5.32 tok/s.
    * **Run 3 (0.767B)**: `emb_dim=1216`, `16q/8kv` heads — absolute compile-time RAM ceiling before OOM.
  * These are **not trimmed versions of the 1.092B model**. They are independent architectures with different embedding dimensions and different head count ratios. The CPU results therefore characterise *what the CPU backend can compile and run*, not how the same model performs across backends.
