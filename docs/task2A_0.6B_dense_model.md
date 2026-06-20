# Qwen 0.6B Dense Model Benchmarking

## 📊 Performance Summary (Best Runs)

| Backend | Best Run | Batch Size | Step Time | Throughput | MFU (TFLOPs/s/dev) | Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CPU** | Run 6 | 2 | 146.51s | 6.99 tok/s | 0.026 | 9.1 GB Host RAM |
| **GPU (T4)** | Run 5 | 8 | 5.44s | 756.53 tok/s | 2.839 | 3.33 GB VRAM |
| **TPU (v5e)** | Run 2 | 16 | 0.314s | 26,052.58 tok/s | 97.751 | 3.36 GB HBM |

### Throughput comparison across backends
```text
CPU (1x)       | 7 tok/s
               | ▏
GPU T4 (108x)  | 757 tok/s
               | █▌
TPU v5e (3722x)| 26,053 tok/s
               | ████████████████████████████████████
```

---

## 🔍 Key Interpretations

### 1. Performance Variance Across Backends
* **CPU performance limitations**: Limited by lack of hardware-accelerated matrix multiplication engines (like Tensor Cores or MXUs) and DDR memory bandwidth (which is 10x slower than GPU VRAM/HBM). Additionally, CPU fails to compile Flash Attention (Pallas kernels) and falls back to slow interpreter paths for dot-product attention.
* **GPU vs. TPU**: TPU v5e operates on dedicated matrix multiply units (systolic arrays) with high bandwidth memory (HBM) and native XLA optimizations, enabling an automated Flash Attention compilation that scales throughput to over 26k tokens/sec at 97.75 TFLOPs/s (49.6% MFU). The Tesla T4 is limited by its older architecture and much lower memory bandwidth (~320 GB/s vs. ~819 GB/s).