# Flash-MoE-CUDA

**Framework-Free MoE Inference on a 6 GB NVIDIA GPU via Async SSD Streaming and Hand-Tuned CUDA Kernels**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows_11-blue)](https://github.com/donelucy46/flash-moe-cuda)
[![GPU: RTX 3050](https://img.shields.io/badge/GPU-RTX_3050-76B900)](https://github.com/amanvx/flash-moe-cuda)

A complete **C++/CUDA** inference engine that runs **Qwen3.5-9B** (hybrid Mixture-of-Experts) at interactive speeds on an **RTX 3050 (6 GB VRAM)** with only **16 GB system RAM** — no Python, no ML frameworks, no third-party libraries.

Inspired by the original [Flash-MoE](https://github.com/danveloper/flash-moe) project on Apple Silicon, this is the **NVIDIA + Windows** counterpart: pure hand-tuned CUDA kernels, async OVERLAPPED I/O, double-buffered pinned staging, and the same "Trust the OS" page-cache philosophy.

### Key Features
- Fully custom CUDA kernels (FMA-optimized Q4_K_M matvec, fused SwiGLU, RoPE, MoE combine+residual)
- GatedDeltaNet layers accelerated with cuBLAS SGEMV
- Async SSD → RAM → PCIe → GPU streaming pipeline
- Permanent non-expert weights in VRAM (~1.8 GB)
- Expert weights streamed on-demand (~3.2 GB total at Q4_K_M)
- Zero Python or framework overhead

**Tested & optimized on RTX 3050 6 GB + 16 GB RAM (Windows 11)**

---

### Quick Start
```bash
git clone https://github.com/amanvx/flash-moe-cuda.git
cd flash-moe-cuda
# (build instructions here)
