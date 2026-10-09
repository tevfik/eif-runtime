/**
 * @file main.c
 * @brief EIF-Runtime Standalone CLI Runner (eif-run)
 *
 * Fast on-device inference for BitNet b1.58 ternary language models.
 * Supports:
 * - Hybrid Gated DeltaNet (Qwen3.5)
 * - Decoder-Only Transformers (Docling, SmolLM, LLaMA)
 */

#define _POSIX_C_SOURCE 199309L
#define _DEFAULT_SOURCE

#include "eif_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_usage(const char *prog)
{
    printf("EIF-Runtime Standalone CLI Runner (v%s)\n\n", EIF_RUNTIME_VERSION_STRING);
    printf("Usage:\n");
    printf("  %s <model_path.eifm> [prompt] [max_tokens] [tokenizer.bin] [options]\n\n", prog);
    printf("Options:\n");
    printf("  -p, --prompt <str>       Input prompt text\n");
    printf("  -n, --max-tokens <int>   Maximum new tokens to generate (default: 64)\n");
    printf("  -t, --tokenizer <path>   Path to custom tokenizer.bin\n");
    printf("  -T, --temp <float>       Sampling temperature (default: 0.7, 0.0=argmax)\n");
    printf("  --top-p <float>          Nucleus top-p threshold (default: 0.9)\n");
    printf("  -r, --rep-penalty <val>  Repetition penalty factor (default: 1.15, 1.0=none)\n");
    printf("  --gpu                    Attempt GPU compute acceleration (graceful CPU fallback)\n");
    printf("  --kv-int8                Enable INT8 quantized KV cache (75%% memory reduction)\n\n");
    printf("Examples:\n");
    printf("  %s artifacts/minicpm5/minicpm5_1b_bitnet.eifm \"Explain edge AI\"\n", prog);
    printf("  %s artifacts/granite_docling/granite_docling_bitnet_dense.eifm \"Convert this page\" --kv-int8\n\n", prog);
}

int main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

#ifdef _OPENMP
    setenv("OMP_WAIT_POLICY", "ACTIVE", 0);
    setenv("OMP_PROC_BIND", "close", 0);
    setenv("GOMP_SPINCOUNT", "100000", 0);
    if (!getenv("OMP_NUM_THREADS")) {
        int procs = omp_get_num_procs();
        omp_set_num_threads(procs >= 8 ? 8 : procs);
    }
#endif

    const char *model_path = NULL;
    const char *prompt = "Hello world!";
    int max_tokens = 64;
    const char *tokenizer_path = NULL;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float rep_penalty = 1.15f;
    bool use_gpu = false;
    int kv_type = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0 || strcmp(argv[i], "n") == 0) {
            if (i + 1 < argc) max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 < argc) tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--temp") == 0 || strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 < argc) temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0) {
            if (i + 1 < argc) top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rep-penalty") == 0 || strcmp(argv[i], "--repetition-penalty") == 0) {
            if (i + 1 < argc) rep_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--kv-int8") == 0) {
            kv_type = 1;
        } else if (argv[i][0] != '-') {
            if (!model_path) {
                model_path = argv[i];
            } else if (!prompt) {
                prompt = argv[i];
            } else if (max_tokens == 128) {
                max_tokens = atoi(argv[i]);
            } else if (!tokenizer_path) {
                tokenizer_path = argv[i];
            }
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    printf("=================================================================\n");
    printf("  EIF-Runtime (Edge-Intelligence Fast Inference Engine v%s)\n", EIF_RUNTIME_VERSION_STRING);
    printf("=================================================================\n");
    printf("  • Model File  : %s\n", model_path);

    /* 1. Load model via polymorphic facade */
    eif_llm_t llm;
    int rc = eif_llm_load(&llm, model_path, tokenizer_path, NULL, 0);
    if (rc != 0) {
        fprintf(stderr, "\n[Error] Failed to load model '%s' (code: %d)\n", model_path, rc);
        return 1;
    }

    const char *arch_name = "Unknown";
    if (llm.arch == EIF_LLM_ARCH_SMOLLM2) {
        arch_name = "Decoder-Only Transformer (Granite / SmolLM / LLaMA)";
    } else if (llm.arch == EIF_LLM_ARCH_QWEN35) {
        arch_name = "Hybrid Gated DeltaNet + Full Attention (Qwen3.5)";
    }

    /* Initialize GPU if requested */
    if (use_gpu) {
        eif_gpu_init(&llm.gpu);
        if (eif_gpu_is_available(&llm.gpu)) {
            printf("  • Compute Unit: GPU [%s] (%s)\n", llm.gpu.device_name, llm.gpu.backend_name);
        } else {
            printf("  • Compute Unit: CPU Native [T-MAC / NEON] (GPU requested but unavailable on this device)\n");
        }
    } else {
        printf("  • Compute Unit: CPU Native [T-MAC / NEON]\n");
    }

    printf("  • Architecture: %s\n", arch_name);
    printf("  • Parameters  : Dim=%d, Hidden=%d, Layers=%d, Heads=%d/%d, Vocab=%d\n",
           llm.dim, llm.hidden_dim, llm.n_layers, llm.n_heads, llm.n_kv_heads, llm.vocab_size);
    printf("  • Tokenizer   : %s (%s)\n",
           llm.tokenizer_path[0] ? llm.tokenizer_path : "Auto-detected",
           llm.tokenizer_loaded ? "Loaded" : "Not Found");
    printf("  • KV Cache    : %s\n", (kv_type == 1) ? "INT8 Quantized (4x Bandwidth Compression)" : "FP32 Native");
    printf("  • Sampling    : Temp=%.2f, Top-p=%.2f, RepPenalty=%.2f\n",
           temperature, top_p, rep_penalty);
    printf("=================================================================\n\n");

    /* 2. Configure generation */
    eif_llm_gen_config_t cfg = {
        .max_new_tokens = max_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .repetition_penalty = rep_penalty,
        .eos_token_id = -1,
        .kv_type = kv_type,
        .use_gpu = use_gpu,
    };

    /* Auto-wrap in ChatML for Qwen3.5 instruct models if not already formatted */
    char formatted_prompt[8192];
    const char *prompt_to_use = prompt;
    if (llm.arch == EIF_LLM_ARCH_QWEN35) {
        if (strstr(prompt, "<|im_start|>") == NULL) {
            snprintf(formatted_prompt, sizeof(formatted_prompt),
                     "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
            prompt_to_use = formatted_prompt;
        }
    }

    printf("--- Prompt ---\n%s\n\n", prompt);
    printf("--- Output Streaming ---\n");

    /* 3. Execute streaming generation */
    double t_start = get_time_sec();
    int gen_tokens = eif_llm_generate(&llm, prompt_to_use, &cfg, NULL, NULL);
    double t_end = get_time_sec();

    double elapsed = t_end - t_start;
    double tok_per_sec = (elapsed > 0.0) ? (gen_tokens / elapsed) : 0.0;
    double ms_per_tok = (gen_tokens > 0) ? (elapsed * 1000.0 / gen_tokens) : 0.0;

    printf("\n\n-----------------------------------------------------------------\n");
    printf("  Performance Summary:\n");
    printf("  • Generated Tokens : %d tokens\n", gen_tokens);
    printf("  • Total Time       : %.3f seconds (wall-clock)\n", elapsed);
    printf("  • Latency          : %.2f ms / token\n", ms_per_tok);
    printf("  • Throughput       : %.1f tokens/sec\n", tok_per_sec);
    printf("-----------------------------------------------------------------\n");

    eif_llm_free(&llm);
    return 0;
}
