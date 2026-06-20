# LLM Training Benchmarking & Optimization (MaxText)

This repository contains the benchmarking results, architecture configurations, and written analysis for training dense and Mixture of Experts (MoE) model architectures using Google's JAX-native MaxText compiler framework across CPU, GPU, and TPU hardware backends.

---

## 📊 Consolidated Benchmarking Dashboard

The table below summarizes the best-performing runs across all combinations of hardware backends, model scales, and architectures.

| Model / Architecture | Backend | Best Run | Model Size | Batch Size | Step Time | Throughput | MFU (TFLOPs/s/dev) | Memory Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen Dense** | **CPU** | Run 6 | 0.596B | 2 | 145.69s | 7.03 tok/s | 0.026 | 9.1 GB Host RAM |
| | **GPU (T4)** | Run 5 | 0.596B | 8 | 5.50s | 744.26 tok/s | 2.792 | 3.33 GB VRAM |
| | **TPU (v5e)** | Run 3 | 0.596B | 24 | 0.464s | 26,492.12 tok/s | 99.400 | 3.36 GB HBM |
| **Qwen Scaled** | **CPU** | Run 2 | 0.732B | 1 | 96.18s | 5.32 tok/s | 0.024 | 9.6 GB Host RAM |
| | **GPU (T4)** | Run 6 | 1.092B | 2 | 2.41s | 425.71 tok/s | 2.865 | 6.10 GB VRAM |
| | **TPU (v5e)** | Run 5 | 1.092B | 12 | 0.373s | 16,485.42 tok/s | 110.932 | 6.13 GB HBM |
| **DeepSeek MoE**| **CPU** | Run 2 | 0.719B | 2 | 142.65s | 7.18 tok/s | 0.007 | 9.1 GB Host RAM |
| | **GPU (T4)** | Run 2 | 0.719B | 2 | 5.41s | 189.27 tok/s | 0.173 | 4.02 GB VRAM |
| | **TPU (v5e)** | Run 5 | 0.719B | 8 | 0.112s | 36,704.15 tok/s | 33.585 | 4.08 GB HBM |

Table values are taken from the final `completed step: 49` metric line of the selected raw training log. Best runs are selected by highest final logged throughput for each model/backend.

---

## 🔍 Consolidated Technical Analysis

### 1. Model Scaling & Constraints (Qwen 0.6B vs. Qwen Scaled 1.09B)
* **TPU MFU by model size**: Scaling the Qwen model architecture from 0.596B to 1.092B parameters resulted in a **~12% improvement in logged TFLOPs/s/device** on the Cloud TPU best-throughput runs (rising from 99.40 to 110.93 TFLOPs/s/dev). This occurs because scaling `base_emb_dim` from 1024 to 1536 aligns the matrix shape more closely with the 128x128 systolic Multiply Unit (MXU) bounds on TPU v5e, reducing padding overhead. Both values are within the theoretical 197 TFLOPs/s peak of TPU v5e.
* **CPU Memory Wall**: While GPU and TPU could scale to 1.092B parameters, CPU training hit a compilation memory wall. Standard XLA graph compilation for a 1.092B model on CPU exceeds 12 GB Host RAM and triggers system OOM crashes. The CPU model was constrained to **0.732B parameters** for execution, highlighting that CPU scaling bottlenecks are dominated by compile-time RAM footprint rather than run-time memory.

### 2. Dense vs. Sparse MoE Performance
* **MoE throughput vs dense**: The DeepSeek MoE model (719M total parameters) achieved **36,704.15 tok/s** on TPU (Run 5) - performing **~38.6% faster** than the dense Qwen 0.6B model (26,492.12 tok/s) and **~2.23x higher** throughput than the Qwen Scaled 1.092B model (16,485.42 tok/s). By dynamically routing tokens to a subset of experts (Top-3 out of 32, plus 1 shared expert), MoE achieves the capacity of a ~719M model while performing calculations equivalent to a much smaller dense model.
* **MFU Discrepancy**: The compute utilization metrics (MFU) for MoE runs appear low (33.585 TFLOPs/s on TPU vs. 110.932 TFLOPs/s for Qwen Scaled). This is a metric calculation artifact: standard MFU calculations assume all model parameters are computed during every forward pass. In MoE, since only the active experts are processed, the actual hardware math utilization is high, but the normalized parameter-based MFU appears artificially depressed.

### 3. Unexpected Findings & Gotchas
* **Pallas CPU Incompatibilities**: The JAX-native Flash Attention Pallas kernels default to GPU/TPU compilation paths and crash on CPU. Bypassing CPU compile errors requires explicitly changing the attention override flag to `attention=dot_product` to bypass custom kernel compilation.
* **Static Memory Layouts**: Across batch scaling runs on GPU and TPU, active device memory allocations (VRAM and HBM) were mostly flat. This is due to JAX's static memory pre-allocation model and XLA's static graph compilation. The compiler reserves static intermediate arrays and memory workspaces during step 1, masking many changes in active batch execution profiles.
* **Colab Environment Conflicts**: Default JAX runtime versions in Colab conflict with MaxText requirements. Upgrading the local virtual environment packages (specifically CUDA-pjrt plugins) and setting `JAX_PLATFORMS=gpu` is required before launching multi-backend pre-training scripts.

---

## 📂 Repository Structure
* [/configs](./configs/): Contains configuration overrides for the best-performing model runs across CPU, GPU, and TPU.
* [/docs](./docs/): Task-specific benchmark metrics, progressions, and individual writeups.
* [/Logs](./Logs/): Raw step logs showing loss progression, compile timings, and device memory allocations.

---

## ⚠️ A Note on Comparisons & Limitations

There are several places in this repository where a strictly apples-to-apples comparison has not been made — and I am fully aware of that. The most prominent example: the "best run" for each model was selected by tuning the batch size independently per backend. This means the headline throughput numbers across models are often achieved at different batch sizes, making direct cross-model comparisons technically imprecise. I made those comparisons anyway for the sake of fulfilling the assignment's objective of benchmarking and contrasting architectures, but they should be read with that caveat in mind.

Similarly, I know that better-performing configurations could have been found — the MoE model on GPU being the clearest example, where the throughput result is a regression relative to dense models due to rematerialization overhead. With more time and familiarity with the framework, those results could likely be meaningfully improved.

---

## 🙏 Closing Note

I came into this assignment as a complete beginner to MaxText, JAX, and LLM pre-training. Everything here — from understanding XLA compilation constraints to debugging Pallas kernel crashes to designing a scaled MoE architecture — was learned from first principles over the last four days.

There is a lot more left to learn, and I know this work is not perfect. But if anything here falls short of your expectations — in depth, accuracy, or results quality — I ask for one more chance. Give me another assignment, and I will prove what I am capable of. I am a fast learner, and I do not give up.

Thank you for the opportunity.

