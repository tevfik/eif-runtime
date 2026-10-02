# EIF-Runtime ⚡

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/C-C99-green.svg)](https://en.wikipedia.org/wiki/C99)
[![Speed](https://img.shields.io/badge/Throughput-150%2B%20tokens%2Fs-orange.svg)]()
[![Precision](https://img.shields.io/badge/BitNet-1.58--bit%20Ternary-purple.svg)]()

**EIF-Runtime** is an ultra-fast, zero-dependency, zero-heap standalone C99 inference engine designed for running **BitNet b1.58 {-1, 0, +1} ternary** and quantized language models on microcontrollers, embedded Linux (Raspberry Pi), and servers.

---

## 🌟 Key Features

* **Multiplication-Free Inference:** Replaces heavy matrix multiplication GEMM with simple addition and subtraction via AVX2 / AVX-512 and ARM NEON kernels.
* **Dual-Architecture Support via Polymorphic API (`eif_llm_t`):**
  * **Hybrid Gated DeltaNet + Full Attention:** Qwen3.5 (89+ tok/s decode on CPU).
  * **Decoder-Only Transformer:** IBM Granite Docling, SmolLM2, LLaMA (150+ tok/s).
* **Zero External Dependencies:** Written in pure, ISO-compliant C99. No PyTorch, no CUDA, no BLAS libraries needed.
* **High Efficiency:** Low memory footprint (< 150 MB for 258M models), sub-10ms token generation latency.
* **OpenMP Parallel Scaling:** Automatic thread pinning and active spinning for consistent peak hardware utilization.

---

## 🚀 Quickstart

### 1. Build from Source
```bash
git clone https://github.com/tevfik/eif-runtime.git
cd eif-runtime
cmake -B build
cmake --build build -j
```

This compiles:
- `libeif_runtime.a` (Static library)
- `libeif_runtime.so` (Shared library)
- `eif-run` (Standalone CLI runner)
- `eif-quickstart` (Minimal embed sample)

### 2. Run CLI Inference
```bash
# Run model with streaming output
./build/eif-run model.eifm "Hello world!" 64
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
        .eos_token_id = -1,
    };

    // 3. Generate autoregressively
    eif_llm_generate(&llm, "What is edge intelligence?", &cfg, token_callback, NULL);

    // 4. Release resources
    eif_llm_free(&llm);
    return 0;
}
```

---

## 📊 Benchmark Comparison

| Model | Architecture | Precision | Binary Size | Throughput (CPU AVX2) | Latency / Token |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Granite Docling 258M** | Transformer | BitNet 1.58b | **143.85 MB** | **156.6 tok/s** | **6.38 ms** |
| **Qwen3.5 0.5B** | Gated DeltaNet | BitNet 2.00b | **229.59 MB** | **89.30 tok/s** | **11.20 ms** |
| **SmolVLM2 256M** | Transformer | BitNet 1.58b | **81.20 MB** | **165.8 tok/s** | **6.03 ms** |

---

## 📂 Repository Structure

```text
eif-runtime/
├── include/
│   ├── eif_runtime.h          # Umbrella master header
│   ├── eif_llm.h              # Unified polymorphic LLM facade
│   ├── eif_qwen35.h           # DeltaNet linear attention engine
│   ├── eif_tinyllm.h          # Transformer BitNet engine
│   ├── eif_bpe_tokenizer.h    # Fast binary BPE tokenizer
│   ├── eif_quantize_bitnet.h  # Multiplication-free ternary kernels
│   └── eif_status.h           # Standard error codes
├── src/
│   ├── eif_llm.c              # Dynamic format & architecture dispatch
│   ├── eif_qwen35.c           # Qwen3.5 forward pass
│   ├── eif_tinyllm.c          # Transformer forward pass
│   ├── eif_bpe_tokenizer.c    # Fast BPE tokenizer decode
│   └── eif_quantize_bitnet.c  # AVX2/AVX-512 BitNet ternary kernels
├── cli/
│   └── main.c                 # Standalone eif-run CLI runner
├── examples/
│   └── quickstart.c           # Minimal C embed application
├── CMakeLists.txt
├── LICENSE
└── README.md
```

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
Created by Tevfik Kadioglu.
