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
* **Hardware-Accelerated INT8 Embedding Engine (`eif_bert`):** Zero-copy mmap GGUF embedding inference supporting Byte-Level BPE (`Ġ`), SentencePiece (` `), and WordPiece (`##`) tokenizers with 0 `<unk>` loss. Native hardware INT8 Dot-Product GEMM (`vdotq_s32` on ARM64, AVX2 / AVX-VNNI on x86) outperforms `llama.cpp` CPU by up to **4.1x–4.5x** (down to **1.28 ms** per embedding).
* **Auto-Detected Pooling & Matryoshka Parity:** Automatic extraction of GGUF `bert.pooling_type` for both CLS Pooling (Granite-30M) and Mean Pooling (MiniLM, Multilingual-E5) with L2 unit normalization.
* **Batched Corpus Embedding API:** High-throughput `eif_bert_embed_batch` for scalable RAG vector database indexing.
* **Resilient Vulkan GPU Compute Fallback:** Optional Vulkan compute shader acceleration for mobile and edge GPUs. If Vulkan is unavailable on headless or baremetal ARM Linux targets, compilation and runtime safely and gracefully fall back to native CPU T-MAC / NEON with zero dependency errors.
* **Native Multimodal Document Understanding (Docling VLM):** Pure C99 document conversion, OCR, and markdown extraction powered by IBM Granite Docling. Features native BitNet b1.58 SubLN RMSNorm scales, SigLIP2 visual token injection, and dynamic multi-tile interleaving without Python or PyTorch runtime dependencies.
* **Causal LLM & BitNet Embedding Engine:** Native text embedding generation (`eif_llm_embed`) supporting **Last-Token Pooling** (`EIF_LLM_POOL_LAST`), Mean Pooling, and L2 unit normalization for causal decoder architectures (Qwen3, BitNet-Embedding).
* **Native GGUF Causal LLM & Embedded Tokenizer:** Direct support for loading GGUF container format decoder models (Qwen2/2.5, LLaMA, Granite, BitNet b1.58) and automatically extracting embedded BPE vocabularies (`tokenizer.ggml.tokens`) for 100% self-contained inference without external `tokenizer.bin` files.
* **Dual-Architecture Support via Polymorphic API (`eif_llm_t`):**
  * **Hybrid Gated DeltaNet + Full Attention:** Qwen3.5 (89+ tok/s decode on CPU).
  * **Decoder-Only Transformer:** IBM Granite Docling, SmolLM2, LLaMA (155+ tok/s).
* **Zero External Dependencies:** Written in pure, ISO-compliant C99. No PyTorch, no CUDA, no BLAS libraries needed.

---

## 📥 Model Zoo (Hugging Face)

This repository contains **zero binary model weights** to keep the source clean, lightweight, and auditable. Pre-converted `.eifm` and `.gguf` models are hosted on Hugging Face:

👉 **[huggingface.co/tevfikk](https://huggingface.co/tevfikk)**

```bash
# Example: Download Granite Docling 258M vision model
pip install huggingface_hub
huggingface-cli download tevfikk/granite-docling-258M-bitnet --local-dir models/granite-docling
```

---

## 🚀 Quickstart

### 1. Build from Source

You can build **EIF-Runtime** in either **Pure CPU Mode** (zero GPU dependencies) or **Vulkan GPU Mode**:

#### A. Pure CPU Mode (No GPU / Zero External Dependencies)
Use this mode if your system does not have a GPU, if you are running in minimal Docker containers or embedded headless Linux (Raspberry Pi, ARM Cortex), or if you do not have Vulkan libraries installed:
```bash
git clone https://github.com/tevfik/eif-runtime.git
cd eif-runtime
cmake -B build -DEIF_ENABLE_VULKAN=OFF
cmake --build build -j
```
* **No GPU libraries or headers required** — 100% ISO C99 + OpenMP.
* Uses vectorized multi-core CPU SIMD kernels (AVX-512, AVX2, ARM NEON).

#### B. Vulkan GPU Compute Mode (Optional Hardware Acceleration)
Use this mode to enable GPU acceleration via cross-vendor Vulkan 1.1 compute shaders (Intel, AMD, NVIDIA, Apple Silicon / MoltenVK, Adreno, Mali):
```bash
cmake -B build -DEIF_ENABLE_VULKAN=ON
cmake --build build -j
```
* **Precompiled SPIR-V shaders** are embedded directly into the C source code as headers — zero runtime or build-time shader compilers (like `glslc` or `glslangValidator`) needed!
* **Automatic Graceful Fallback:** Even when compiled with `-DEIF_ENABLE_VULKAN=ON`, if no physical GPU or Vulkan ICD driver is found at runtime, EIF-Runtime automatically and safely falls back to CPU SIMD execution with zero errors.

This compiles:
- `libeif_runtime.a` (Static library)
- `libeif_runtime.so` (Shared library)
- `eif-run` (Standalone CLI runner)

### 2. Run CLI Inference

#### A. Multimodal Document Parsing (Granite-Docling VLM)
```bash
# Parse a document image (invoice, receipt, CV, article) directly to structured markdown:
./build/eif-run models/granite-docling/model.eifm "Convert this document to markdown." \
    --image /path/to/document.png \
    --max-tokens 512
```

#### B. Pure Text Generation (Granite / SmolLM2 / Qwen3.5)
```bash
# Standard generative inference
./build/eif-run model.eifm -p "Hello world!" --max-tokens 64

# Maximum efficiency: T-MAC + INT8 KV Cache + Repetition Penalty
./build/eif-run model.eifm -p "Explain edge AI" --max-tokens 64 --kv-int8 -r 1.15

# Request GPU acceleration (auto falls back to CPU if unavailable)
./build/eif-run model.eifm -p "Document analysis:" --max-tokens 64 --gpu --kv-int8
```

#### C. Dense Semantic Vector Embeddings (BERT / Granite / Multilingual-E5)
```bash
# IBM Granite-30M (Byte-Level BPE, CLS pooling, 384-dim, ~1.28 ms on CPU):
./build/eif-run models/dist/granite-30m-q8_0.gguf --embed "Artificial intelligence and edge computing."

# all-MiniLM-L6-v2 (SentencePiece / WordPiece, Mean pooling, 384-dim, ~1.50 ms on CPU):
./build/eif-run models/dist/minilm.gguf -e "Semantic vector similarity search"

# Multilingual-E5-Small (SentencePiece, Multilingual Turkish/English, Mean pooling):
./build/eif-run models/dist/multilingual-e5-small-q8_0.gguf -e "query: Kenar yapay zeka çıkarım motoru."
```

#### D. Standalone CLI Options Reference
| Option | Description | Default | Applicable Modes |
| :--- | :--- | :---: | :--- |
| `-p`, `--prompt <str>` | Input text prompt for generative LLM | `"Hello world!"` | Generative / VLM |
| `-e`, `--embed <str>` | Input text for dense vector embedding extraction | None | Embedding (`eif_bert`) |
| `-i`, `--image <path>` | Input document image for vision multimodal OCR / conversion | None | Multimodal Docling |
| `-n`, `--max-tokens <N>` | Maximum new tokens to generate | `64` | Generative / VLM |
| `-t`, `--tokenizer <path>` | Path to external `tokenizer.bin` (auto-detected if omitted) | Auto | Generative LLMs |
| `-T`, `--temp <val>` | Sampling temperature (`0.0` for deterministic / greedy) | `0.7` | Generative LLMs |
| `--top-p <val>` | Nucleus sampling probability cutoff threshold | `0.9` | Generative LLMs |
| `-r`, `--rep-penalty <val>` | Repetition penalty factor (`1.0` = none) | `1.15` | Generative LLMs |
| `--gpu` | Attempt Vulkan GPU compute acceleration (graceful CPU fallback) | Disabled | All models |
| `--kv-int8` | Enable INT8 quantized KV-cache (4x memory reduction) | Disabled (FP32) | Generative LLMs |

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

### Multimodal Document Parsing (Docling VLM)
```c
// Parse document image directly with IBM Granite Docling:
eif_llm_t docling;
if (eif_llm_load(&docling, "granite_docling_vision.eifm", NULL, NULL, 0) == 0) {
    eif_llm_gen_config_t cfg = {
        .max_new_tokens = 512,
        .temperature = 0.0f, // Greedy decoding for high-accuracy OCR / Markdown
    };
    eif_llm_generate_vision_docling(&docling, "document.png", "Convert this document to markdown.", &cfg, token_callback, NULL);
    eif_llm_free(&docling);
}
```

### Dense Vector Embeddings (BERT / Granite / Multilingual-E5)
```c
#include "eif_bert.h"

int main(void) {
    eif_bert_t bert;
    // Load GGUF format embedding model (e.g. Granite-30M Q8_0)
    if (eif_bert_load(&bert, "models/dist/granite-30m-q8_0.gguf") != 0) return 1;

    // 1. Single sentence embedding (CLS or Mean pooling auto-selected)
    float embedding[384];
    eif_bert_embed(&bert, "Artificial intelligence on edge hardware.", embedding);

    // 2. Batched corpus embedding for RAG vector database indexing
    const char *documents[] = {
        "First document paragraph text.",
        "Second document paragraph text."
    };
    float batch_embeddings[2 * 384];
    eif_bert_embed_batch(&bert, documents, 2, batch_embeddings);

    eif_bert_free(&bert);
    return 0;
}
```

---

## 📊 Benchmark Comparison

### 1. Generative BitNet b1.58 Models
Evaluated on ARM Cortex (Single Core, Wall-Clock):

| Model & Engine | Precision | KV-Cache Mode | Latency / Token | Throughput | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Granite Docling 258M (NEON Bitmask Baseline)** | BitNet 1.58b | FP32 Native | 18.50 ms | 54.0 tok/s | 1.00x |
| **Granite Docling 258M (T-MAC LUT)** | BitNet 1.58b | FP32 Native | 9.92 ms | 100.8 tok/s | 1.86x |
| **Granite Docling 258M (T-MAC + INT8 KV)** | BitNet 1.58b | **INT8 Quantized** | **6.43 ms** | **155.5 tok/s** | **2.88x 🚀** |
| **Granite Docling 258M (128 Tokens Sustained)** | BitNet 1.58b | **INT8 Quantized** | **9.05 ms** | **110.5 tok/s** | **2.05x** |
| **Qwen3.5 0.5B (Hybrid DeltaNet)** | BitNet 2.00b | FP32 Native | 11.20 ms | 89.3 tok/s | 1.65x |
| **SmolVLM2 256M (Transformer)** | BitNet 1.58b | **INT8 Quantized** | **6.03 ms** | **165.8 tok/s** | **3.07x 🚀** |

### 2. Dense Semantic Embedding Models: EIF-Runtime vs llama.cpp
Evaluated on ARM64 Cortex Architecture (Wall-Clock, 4 Threads):

| Model & Parameters | Format | Inference Engine | Hardware Backend | Latency / Sentence | Throughput | vs llama.cpp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all-MiniLM-L6-v2 (22M)** | Q8_0 | `llama.cpp` | CPU (NEON) | 6.22 ms | 160.8 emb/s | 1.00x |
| **all-MiniLM-L6-v2 (22M)** | Q8_0 | **EIF-Runtime (Previous)** | CPU (FMADD) | 16.47 ms | 60.7 emb/s | 0.38x |
| **all-MiniLM-L6-v2 (22M)** | Q8_0 | **EIF-Runtime (Accelerated)** | **CPU (INT8 `vdotq`)** | **1.50 ms** | **666.3 emb/s** | **4.15x Faster 🚀** |
| **Granite-30M (30M)** | Q8_0 | `llama.cpp` | CPU (NEON) | 5.80 ms | 172.4 emb/s | 1.00x |
| **Granite-30M (30M)** | Q8_0 | **EIF-Runtime (Accelerated)** | **CPU (INT8 `vdotq`)** | **1.28 ms** | **779.9 emb/s** | **4.53x Faster 🚀** |
| **Multilingual-E5-Small (118M)** | Q8_0 | `llama.cpp` | CPU (NEON) | 18.40 ms | 54.3 emb/s | 1.00x |
| **Multilingual-E5-Small (118M)** | Q8_0 | **EIF-Runtime (Accelerated)** | **CPU (INT8 `vdotq`)** | **5.55 ms** | **180.1 emb/s** | **3.31x Faster 🚀** |
| **MiniLM-L6-v2 (Batched B=64)** | Q8_0 | **EIF-Runtime (Vulkan GPU)** | **Vulkan Compute** | **0.42 ms / emb** | **2,380 emb/s** | **14.8x Faster 🚀** |

> **Key Architectural Takeaways:**
> - **CPU Efficiency:** For real-time on-device edge queries (Batch size = 1, $T \le 64$), **EIF-Runtime is 3.3x–4.5x faster than `llama.cpp` on CPU** because it utilizes zero-heap preallocated scratch spaces, eliminates graph scheduling overhead, and executes unified INT8 dot-product instructions (`vdotq_s32` on ARM, AVX2 `madd` on x86).
> - **Vulkan GPU Acceleration:** For large document indexing and batch inference ($B \ge 64$), Vulkan compute shaders unlock over 2,300+ embeddings/sec.

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
