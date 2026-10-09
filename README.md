# EIF-Runtime ⚡

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/C-C99-green.svg)](https://en.wikipedia.org/wiki/C99)
[![Speed](https://img.shields.io/badge/Throughput-155%2B%20tokens%2Fs-orange.svg)]()
[![Precision](https://img.shields.io/badge/BitNet-1.58--bit%20Ternary-purple.svg)]()
[![Backend](https://img.shields.io/badge/Compute-T--MAC%20LUT%20%2B%20Vulkan-brightgreen.svg)]()

**EIF-Runtime** is an ultra-fast, zero-dependency, zero-heap standalone C99 inference engine designed for running **BitNet b1.58 {-1, 0, +1} ternary** and quantized language & vision-language models on microcontrollers, embedded Linux (Raspberry Pi, ARM Cortex), and servers.

---

## 🌟 Key Features

* **T-MAC Table-Lookup Matrix Engine (LUT):** Replaces multiplications with amortized activation LUT caching and branchless table lookups, delivering a **1.8x–2.1x speedup** over traditional bitmask NEON kernels.
* **INT8 Activation & Pure Integer Arithmetic:** Full support for BitNet b1.58 activation quantization with dynamic absmax scaling ($I8 \times \text{Ternary}$). Inner loops run purely on integer additions/subtractions with zero floating-point operations.
* **Sub-Byte Block Quantization:** Fine-grained block scaling (32, 64, or 128 elements per block, matching `llama.cpp` `Q2_K` / `IQ2`) to eliminate outlier-induced quantization degradation.
* **INT8 Quantized KV-Cache:** Head-wise dynamic INT8 quantization for Key-Value caches (`--kv-int8`), reducing memory consumption and attention memory bandwidth by **75% (4x compression)**.
* **Resilient Vulkan GPU Compute Fallback:** Optional Vulkan compute shader acceleration for mobile and edge GPUs. If Vulkan is unavailable on headless or baremetal ARM Linux targets, compilation and runtime safely and gracefully fall back to native CPU T-MAC / NEON with zero dependency errors.
* **Causal LLM & BitNet Embedding Engine:** Native text embedding generation (`eif_llm_embed`) supporting **Last-Token Pooling** (`EIF_LLM_POOL_LAST`), Mean Pooling, and L2 unit normalization for causal decoder architectures (Qwen3, BitNet-Embedding).
* **Native GGUF Causal LLM & Embedded Tokenizer:** Direct support for loading GGUF container format decoder models (Qwen2/2.5, LLaMA, Granite, BitNet b1.58) and automatically extracting embedded BPE vocabularies (`tokenizer.ggml.tokens`) for 100% self-contained inference without external `tokenizer.bin` files.
* **Dual-Architecture Support via Polymorphic API (`eif_llm_t`):**
  * **Hybrid Gated DeltaNet + Full Attention:** Qwen3.5 (89+ tok/s decode on CPU).
  * **Decoder-Only Transformer:** IBM Granite Docling, SmolLM2, LLaMA (155+ tok/s).
* **Zero External Dependencies:** Written in pure, ISO-compliant C99. No PyTorch, no CUDA, no BLAS libraries needed.

---

## 🚀 Quickstart

### 1. Build from Source
```bash
git clone https://github.com/tevfik/eif-runtime.git
cd eif-runtime
cmake -B build
cmake --build build -j
```

> **Note on GPU Acceleration:** To compile with optional Vulkan GPU support:
> ```bash
> cmake -B build -DEIF_ENABLE_VULKAN=ON
> cmake --build build -j
> ```
> If Vulkan headers or drivers are absent, CMake safely defaults to pure CPU execution without breaking the build.

This compiles:
- `libeif_runtime.a` (Static library)
- `libeif_runtime.so` (Shared library)
- `eif-run` (Standalone CLI runner)

### 2. Run CLI Inference
```bash
# Standard inference
./build/eif-run model.eifm "Hello world!" 64

# Maximum efficiency: T-MAC + INT8 KV Cache + Repetition Penalty
./build/eif-run model.eifm "Explain edge AI" 64 --kv-int8 -r 1.15

# Request GPU acceleration (auto falls back to CPU if unavailable)
./build/eif-run model.eifm "Document analysis:" 64 --gpu --kv-int8
```

---

## 💻 C API Integration

Embedding `eif-runtime` into any C/C++ application requires less than 20 lines of code:

```c
#include "eif_runtime.h"
#include <stdio.h>

void token_callback(const char *piece, int32_t token_id, void *user_data) {
    if (piece) {
        fputs(piece, stdout);
        fflush(stdout);
    }
}

int main(void) {
    eif_llm_t llm;
    // 1. Load model (.eifm format)
    if (eif_llm_load(&llm, "model.eifm", NULL, NULL, 0) != 0) {
        return 1;
    }

    // 2. Configure generation sampling
    eif_llm_gen_config_t cfg = {
        .max_new_tokens = 64,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .repetition_penalty = 1.15f,
        .kv_type = 1, // INT8 Quantized KV Cache
        .eos_token_id = -1,
    };

    // 3. Generate autoregressively
    eif_llm_generate(&llm, "What is edge intelligence?", &cfg, token_callback, NULL);

    // 4. Release resources
    eif_llm_free(&llm);
    return 0;
}
```

### Text Embeddings with Last-Token Pooling
```c
// Extract 576-dim or 1024-dim normalized embedding vector from Causal LLM:
float embedding[1024];
eif_llm_embed(&llm, "Document query string", embedding, EIF_LLM_POOL_LAST);
```

---

## 📊 Benchmark Comparison

Evaluated on ARM Cortex-X925 (Single Core, Wall-Clock):

| Model & Engine | Precision | KV-Cache Mode | Latency / Token | Throughput | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Granite Docling 258M (NEON Bitmask Baseline)** | BitNet 1.58b | FP32 Native | 18.50 ms | 54.0 tok/s | 1.00x |
| **Granite Docling 258M (T-MAC LUT)** | BitNet 1.58b | FP32 Native | 9.92 ms | 100.8 tok/s | 1.86x |
| **Granite Docling 258M (T-MAC + INT8 KV)** | BitNet 1.58b | **INT8 Quantized** | **6.43 ms** | **155.5 tok/s** | **2.88x 🚀** |
| **Granite Docling 258M (128 Tokens Sustained)** | BitNet 1.58b | **INT8 Quantized** | **9.05 ms** | **110.5 tok/s** | **2.05x** |
| **Qwen3.5 0.5B (Hybrid DeltaNet)** | BitNet 2.00b | FP32 Native | 11.20 ms | 89.3 tok/s | 1.65x |
| **SmolVLM2 256M (Transformer)** | BitNet 1.58b | **INT8 Quantized** | **6.03 ms** | **165.8 tok/s** | **3.07x 🚀** |

---

## 📂 Repository Structure

```text
eif-runtime/
├── include/
│   ├── eif_runtime.h          # Umbrella master header
│   ├── eif_llm.h              # Unified polymorphic LLM & embedding facade
│   ├── eif_gguf.h             # Fast C99 GGUF container parser
│   ├── eif_gpu.h              # Optional Vulkan GPU backend & CPU fallback
│   ├── eif_bert.h             # High-performance BERT embedding engine
│   ├── eif_qwen35.h           # DeltaNet linear attention engine
│   ├── eif_tinyllm.h          # Transformer BitNet engine with INT8 KV
│   ├── eif_bpe_tokenizer.h    # Fast binary BPE tokenizer
│   ├── eif_quantize_bitnet.h  # T-MAC LUT & multiplication-free ternary kernels
│   └── eif_status.h           # Standard error codes
├── src/
│   ├── eif_llm.c              # Dynamic format & architecture dispatch
│   ├── eif_gpu.c              # Vulkan compute pipeline & CPU stubs
│   ├── eif_bert.c             # GGUF BERT engine & WordPiece/SPM tokenizer
│   ├── eif_qwen35.c           # Qwen3.5 forward pass
│   ├── eif_tinyllm.c          # Transformer forward pass with INT8 KV
│   ├── eif_bpe_tokenizer.c    # Fast BPE tokenizer decode
│   └── eif_quantize_bitnet.c  # T-MAC, ARM NEON & AVX2 BitNet kernels
├── cli/
│   └── main.c                 # Standalone eif-run CLI runner (--gpu, --kv-int8)
├── CMakeLists.txt
├── LICENSE
└── README.md
```

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).  
Created by Tevfik Kadioglu.
