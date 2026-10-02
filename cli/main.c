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
    printf("  %s <model_path.eifm> [prompt] [max_tokens] [tokenizer.bin]\n\n", prog);
    printf("Examples:\n");
    printf("  %s granite_docling_bitnet_dense.eifm \"Convert this document to markdown.\"\n", prog);
    printf("  %s qwen35_bitnet_2bit_e2bit.eifm \"Hello, who are you?\" 64\n\n", prog);
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

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) {
            if (i + 1 < argc) max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 < argc) tokenizer_path = argv[++i];
        } else if (argv[i][0] != '-') {
            if (!model_path) {
                model_path = argv[i];
            } else if (prompt == NULL || strcmp(prompt, "Hello world!") == 0) {
                prompt = argv[i];
            } else if (max_tokens == 64) {
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

    printf("  • Architecture: %s\n", arch_name);
    printf("  • Parameters  : Dim=%d, Hidden=%d, Layers=%d, Heads=%d/%d, Vocab=%d\n",
           llm.dim, llm.hidden_dim, llm.n_layers, llm.n_heads, llm.n_kv_heads, llm.vocab_size);
    printf("  • Tokenizer   : %s (%s)\n",
           llm.tokenizer_path[0] ? llm.tokenizer_path : "Auto-detected",
           llm.tokenizer_loaded ? "Loaded" : "Not Found");
    printf("=================================================================\n\n");

    /* 2. Configure generation */
    eif_llm_gen_config_t cfg = {
        .max_new_tokens = max_tokens,
        .temperature = 0.0f,  /* 0.0 = greedy argmax */
        .top_p = 0.9f,
        .eos_token_id = -1,
    };

    printf("--- Prompt ---\n%s\n\n", prompt);
    printf("--- Output Streaming ---\n");

    /* 3. Execute streaming generation */
    double t_start = get_time_sec();
    int gen_tokens = eif_llm_generate(&llm, prompt, &cfg, NULL, NULL);
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
