# Qwen 0.6B Dense Model Benchmarking

## 📊 Performance Summary (Best Runs)

| Backend | Best Run | Batch Size | Step Time | Throughput | MFU (TFLOPs/s/dev) | Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CPU** | Run 6 | 2 | 145.69s | 7.03 tok/s | 0.026 | 9.1 GB Host RAM |
| **GPU (T4)** | Run 5 | 8 | 5.50s | 744.26 tok/s | 2.792 | 3.33 GB VRAM |
| **TPU (v5e)** | Run 3 | 24 | 0.464s | 26,492.12 tok/s | 99.400 | 3.36 GB HBM |

Table values are taken from the final `completed step: 49` metric line of the selected raw training log. Best runs are selected by highest final logged throughput for each backend.

### Throughput comparison across backends
```text
CPU (1x)       | 7 tok/s
               | ▏
GPU T4 (106x)  | 744 tok/s
               | █▌
TPU v5e (3770x)| 26,492 tok/s
               | ████████████████████████████████████
```

---

## 🔍 Key Interpretations

### 1. Performance Variance Across Backends
* **CPU performance limitations**: Limited by lack of hardware-accelerated matrix multiplication engines (like Tensor Cores or MXUs) and DDR memory bandwidth. CPU fails to compile the default Flash Attention/Pallas path, so the successful CPU runs explicitly use `attention=dot_product`.
* **GPU vs. TPU**: TPU v5e operates on dedicated matrix multiply units (systolic arrays) with high bandwidth memory (HBM) and native XLA optimizations, scaling throughput to over 26k tokens/sec at 99.40 TFLOPs/s on the best-throughput run. The Tesla T4 is limited by its older architecture and much lower memory bandwidth (~320 GB/s vs. ~819 GB/s).
