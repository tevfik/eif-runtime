/**
 * @file eif_llm.c
 * @brief Unified C99 Runtime Implementation for Large Language Models
 */

#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "eif_llm.h"

#include <ctype.h>
#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEFAULT_BUFFER_SIZE (256 * 1024 * 1024) /* 256 MB */

eif_llm_arch_t eif_llm_detect_arch(const char *model_path)
{
    if (!model_path) return EIF_LLM_ARCH_UNKNOWN;

    FILE *f = fopen(model_path, "rb");
    if (!f) return EIF_LLM_ARCH_UNKNOWN;

    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return EIF_LLM_ARCH_UNKNOWN;
    }
    fclose(f);

    /* 0x51574E35 = 'QWN5' */
    if (magic == 0x51574E35) {
        return EIF_LLM_ARCH_QWEN35;
    }
    /* 0x53564C4D = 'SVLM', 0x54544D4C = 'TTML', 0x544C4C4D = 'TLLM' */
    if (magic == 0x53564C4D || magic == 0x54544D4C || magic == 0x544C4C4D) {
        return EIF_LLM_ARCH_SMOLLM2;
    }
    /* 0x54524542 = 'BERT' */
    if (magic == 0x54524542) {
        return EIF_LLM_ARCH_BERT;
    }
    /* 0x46554747 = 'GGUF' */
    if (magic == 0x46554747) {
        return EIF_LLM_ARCH_BERT;
    }

    /* Fallback heuristic by filename */
    if (strstr(model_path, "bert") != NULL || strstr(model_path, "minilm") != NULL) {
        return EIF_LLM_ARCH_BERT;
    }
    if (strstr(model_path, "qwen") != NULL) {
        return EIF_LLM_ARCH_QWEN35;
    }
    if (strstr(model_path, "smol") != NULL || strstr(model_path, "docling") != NULL ||
        strstr(model_path, "granite") != NULL || strstr(model_path, "llama") != NULL ||
        strstr(model_path, "minicpm") != NULL) {
        return EIF_LLM_ARCH_SMOLLM2;
    }

    return EIF_LLM_ARCH_UNKNOWN;
}

size_t eif_llm_compute_buffer_size(const char *model_path)
{
    if (!model_path) return DEFAULT_BUFFER_SIZE;
    FILE *f = fopen(model_path, "rb");
    if (!f) return DEFAULT_BUFFER_SIZE;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    if (sz <= 0) return DEFAULT_BUFFER_SIZE;
    size_t needed = (size_t)sz + (128 * 1024 * 1024);
    return needed > DEFAULT_BUFFER_SIZE ? needed : DEFAULT_BUFFER_SIZE;
}

static bool find_tokenizer_path(const char *model_path, char *out_tok_path, size_t max_len)
{
    if (!model_path || !out_tok_path) return false;

    char dir_buf[512];
    snprintf(dir_buf, sizeof(dir_buf), "%s", model_path);
    char *dir = dirname(dir_buf);

    /* Check in model directory */
    char candidate[512];
    snprintf(candidate, sizeof(candidate), "%s/tokenizer.bin", dir);
    if (access(candidate, F_OK) == 0) {
        snprintf(out_tok_path, max_len, "%s", candidate);
        return true;
    }

    snprintf(candidate, sizeof(candidate), "%s/qwen_tokenizer.bin", dir);
    if (access(candidate, F_OK) == 0) {
        snprintf(out_tok_path, max_len, "%s", candidate);
        return true;
    }

    /* Known global artifact paths */
    const char *defaults[] = {
        "artifacts/minicpm5/tokenizer.bin",
        "artifacts/granite_docling/tokenizer.bin",
        "artifacts/smolvlm2_bitnet/tokenizer.bin",
        "artifacts/qwen35/qwen_tokenizer.bin",
        NULL
    };

    for (int i = 0; defaults[i] != NULL; i++) {
        if (access(defaults[i], F_OK) == 0) {
            snprintf(out_tok_path, max_len, "%s", defaults[i]);
            return true;
        }
    }

    return false;
}

int eif_llm_load(eif_llm_t *llm, const char *model_path, const char *tokenizer_path,
                 void *buffer, size_t buffer_size)
{
    if (!llm || !model_path) return -1;
    memset(llm, 0, sizeof(eif_llm_t));

    snprintf(llm->model_path, sizeof(llm->model_path), "%s", model_path);
    llm->arch = eif_llm_detect_arch(model_path);
    if (llm->arch == EIF_LLM_ARCH_UNKNOWN) {
        fprintf(stderr, "[EIF LLM Error] Unknown model architecture: %s\n", model_path);
        return -2;
    }

    /* Buffer allocation */
    if (buffer && buffer_size > 0) {
        llm->buffer = (uint8_t *)buffer;
        llm->buffer_size = buffer_size;
        llm->owns_buffer = false;
    } else {
        llm->buffer_size = eif_llm_compute_buffer_size(model_path);
        llm->buffer = (uint8_t *)malloc(llm->buffer_size);
        if (!llm->buffer) {
            fprintf(stderr, "[EIF LLM Error] Failed to allocate %zu bytes\n", llm->buffer_size);
            return -3;
        }
        llm->owns_buffer = true;
    }

    /* Architecture-specific backend initialization */
    if (llm->arch == EIF_LLM_ARCH_BERT) {
        if (eif_bert_load(&llm->backend.bert, model_path) != 0) {
            fprintf(stderr, "[EIF LLM Error] Failed to load BERT model: %s\n", model_path);
            if (llm->owns_buffer && llm->buffer) free(llm->buffer);
            return -4;
        }
        llm->dim = llm->backend.bert.config.dim;
        llm->hidden_dim = llm->backend.bert.config.intermediate_dim;
        llm->n_layers = llm->backend.bert.config.n_layers;
        llm->n_heads = llm->backend.bert.config.n_heads;
        llm->n_kv_heads = llm->backend.bert.config.n_heads;
        llm->vocab_size = llm->backend.bert.config.vocab_size;
        llm->seq_len = llm->backend.bert.config.max_seq_len;
        llm->tokenizer_loaded = (llm->backend.bert.vocab.tokens != NULL);
        llm->is_initialized = true;
        return 0;
    } else if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        size_t state_sz = 128 * 1024 * 1024;
        if (qwen35_load(&llm->backend.qwen35, model_path, llm->buffer, state_sz) != 0) {
            fprintf(stderr, "[EIF LLM Error] Failed to load Qwen3.5: %s\n", model_path);
            if (llm->owns_buffer) free(llm->buffer);
            return -4;
        }
        llm->dim = llm->backend.qwen35.config.dim;
        llm->hidden_dim = llm->backend.qwen35.config.hidden_dim;
        llm->n_layers = llm->backend.qwen35.config.n_layers;
        llm->n_heads = llm->backend.qwen35.config.n_heads;
        llm->n_kv_heads = llm->backend.qwen35.config.n_kv_heads;
        llm->vocab_size = llm->backend.qwen35.config.vocab_size;
        llm->seq_len = llm->backend.qwen35.config.max_seq_len;
    } else {
        /* LLaMA / SmolLM2 / Granite-Docling (eif_tinyllm backend) */
        FILE *f = fopen(model_path, "rb");
        if (!f) {
            if (llm->owns_buffer) free(llm->buffer);
            return -5;
        }

        uint32_t magic = 0;
        if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            if (llm->owns_buffer) free(llm->buffer);
            return -5;
        }

        tinyllm_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));

        if (magic == 0x53564C4D) {
            /* SVLM format: 48 bytes header + 128 bytes config (32 x int32) */
            int32_t cfg_ints[32];
            fseek(f, 48, SEEK_SET);
            if (fread(cfg_ints, sizeof(int32_t), 32, f) != 32) {
                fclose(f);
                if (llm->owns_buffer) free(llm->buffer);
                return -6;
            }
            cfg.dim = cfg_ints[0];
            cfg.hidden_dim = cfg_ints[1];
            cfg.n_layers = cfg_ints[2];
            cfg.n_heads = cfg_ints[3];
            cfg.n_kv_heads = cfg_ints[4];
            cfg.head_dim = cfg_ints[5];
            cfg.vocab_size = cfg_ints[6];
            cfg.seq_len = cfg_ints[7] > 0 ? cfg_ints[7] : 512;
            cfg.rope_theta = (float)cfg_ints[8];
            cfg.qtype = (cfg_ints[12] == 3) ? TINYLLM_QTYPE_BITNET_158 : (tinyllm_qtype_t)cfg_ints[12];
        } else {
            /* TTML / TLLM format: 36 bytes header + tinyllm_config_t */
            fseek(f, 36, SEEK_SET);
            if (fread(&cfg, sizeof(cfg), 1, f) != 1) {
                fclose(f);
                if (llm->owns_buffer) free(llm->buffer);
                return -6;
            }
        }
        fclose(f);

        size_t mem_size = tinyllm_memory_size(&cfg);
        if (tinyllm_init(&llm->backend.tinyllm, &cfg, llm->buffer, mem_size) != 0) {
            fprintf(stderr, "[EIF LLM Error] Failed to init TinyLLM state\n");
            if (llm->owns_buffer) free(llm->buffer);
            return -7;
        }

        uint8_t *weights_buf = llm->buffer + mem_size;
        size_t weights_buf_size = llm->buffer_size - mem_size;
        if (tinyllm_load_quantized_model(&llm->backend.tinyllm, model_path, weights_buf, weights_buf_size) != 0) {
            fprintf(stderr, "[EIF LLM Error] Failed to load TinyLLM quantized weights\n");
            if (llm->owns_buffer) free(llm->buffer);
            return -8;
        }

        llm->dim = cfg.dim;
        llm->hidden_dim = cfg.hidden_dim;
        llm->n_layers = cfg.n_layers;
        llm->n_heads = cfg.n_heads;
        llm->n_kv_heads = cfg.n_kv_heads;
        llm->vocab_size = cfg.vocab_size;
        llm->seq_len = cfg.seq_len;
    }

    /* Tokenizer resolution & loading */
    char resolved_tok[512] = {0};
    if (tokenizer_path && access(tokenizer_path, F_OK) == 0) {
        snprintf(resolved_tok, sizeof(resolved_tok), "%s", tokenizer_path);
    } else {
        find_tokenizer_path(model_path, resolved_tok, sizeof(resolved_tok));
    }

    if (strlen(resolved_tok) > 0) {
        snprintf(llm->tokenizer_path, sizeof(llm->tokenizer_path), "%s", resolved_tok);
        if (eif_bpe_tokenizer_load(&llm->tokenizer, resolved_tok, llm->vocab_size) == EIF_STATUS_OK) {
            llm->tokenizer_loaded = true;
        } else {
            fprintf(stderr, "[EIF LLM Warning] Tokenizer failed to load from %s\n", resolved_tok);
        }
    }

    llm->current_pos = 0;
    llm->is_initialized = true;
    return 0;
}

int eif_llm_forward(eif_llm_t *llm, int token, int pos)
{
    if (!llm || !llm->is_initialized) return -1;

    float *logits = NULL;
    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        logits = qwen35_forward(&llm->backend.qwen35, token, pos);
    } else {
        logits = tinyllm_forward(&llm->backend.tinyllm, token, pos);
    }

    llm->current_pos = pos + 1;
    return (logits != NULL) ? 0 : -2;
}

int eif_llm_forward_no_logits(eif_llm_t *llm, int token, int pos)
{
    if (!llm || !llm->is_initialized) return -1;

    float *hidden = NULL;
    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        hidden = qwen35_forward_no_logits(&llm->backend.qwen35, token, pos);
    } else {
        hidden = tinyllm_forward_no_logits(&llm->backend.tinyllm, token, pos);
    }

    llm->current_pos = pos + 1;
    return (hidden != NULL) ? 0 : -2;
}

float* eif_llm_get_logits(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return NULL;
    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        return llm->backend.qwen35.state.logits;
    } else {
        return llm->backend.tinyllm.state.logits;
    }
}

int eif_llm_sample(eif_llm_t *llm, float temperature, float top_p)
{
    float *logits = eif_llm_get_logits(llm);
    if (!logits) return 0;
    int vocab_size = llm->vocab_size;

    /* Greedy argmax */
    if (temperature <= 0.0001f) {
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    /* Softmax with temperature scaling */
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum_exp += logits[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_sum;
    }

    /* Top-p / Random selection */
    float coin = ((float)rand()) / (float)RAND_MAX;
    if (top_p > 0.0f && top_p < 1.0f) {
        coin *= top_p;
    }
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (coin <= cdf) {
            return i;
        }
    }
    return vocab_size - 1;
}

int eif_llm_generate(eif_llm_t *llm, const char *prompt, const eif_llm_gen_config_t *cfg,
                     eif_llm_token_callback_fn cb, void *user_data)
{
    if (!llm || !llm->is_initialized || !prompt) return -1;
    if (!llm->tokenizer_loaded) {
        fprintf(stderr, "[EIF LLM Error] Tokenizer is required for generate()\n");
        return -2;
    }

    eif_llm_gen_config_t conf;
    memset(&conf, 0, sizeof(conf));
    if (cfg) {
        conf = *cfg;
    } else {
        conf.max_new_tokens = 64;
        conf.temperature = 0.7f;
        conf.top_p = 0.9f;
        conf.repetition_penalty = 1.15f;
        conf.eos_token_id = llm->tokenizer.eos_token_id;
        conf.kv_type = 0;
        conf.use_gpu = false;
    }
    if (conf.repetition_penalty <= 0.0f) {
        conf.repetition_penalty = 1.0f;
    }

    if (llm->arch == EIF_LLM_ARCH_SMOLLM2) {
        llm->backend.tinyllm.config.kv_type = conf.kv_type;
    }

    /* Tokenize prompt */
    int32_t prompt_tokens[2048];
    int n_prompt = eif_bpe_tokenizer_encode(&llm->tokenizer, prompt, 0, 0, prompt_tokens, 2048);
    if (n_prompt <= 0) return -3;
    printf("[Tokenizer Debug] Prompt tokens (%d): [", n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        printf("%d%s", prompt_tokens[i], (i < n_prompt - 1) ? ", " : "");
    }
    printf("]\n");

    /* Prefill phase */
    int current_token = prompt_tokens[0];
    for (int i = 0; i < n_prompt; i++) {
        current_token = prompt_tokens[i];
        if (eif_llm_forward(llm, current_token, i) != 0) {
            return -4;
        }
    }

    int32_t history_tokens[4096];
    int total_history = 0;
    for (int i = 0; i < n_prompt && total_history < 4096; i++) {
        history_tokens[total_history++] = prompt_tokens[i];
    }

    int generated_count = 0;
    int pos = n_prompt;

    /* Autoregressive generation phase */
    for (int g = 0; g < conf.max_new_tokens; g++) {
        /* Apply repetition penalty to logits of previously seen tokens */
        if (conf.repetition_penalty > 1.0f && total_history > 0) {
            float *logits = eif_llm_get_logits(llm);
            if (logits) {
                for (int h = 0; h < total_history; h++) {
                    int tid = history_tokens[h];
                    if (tid >= 0 && tid < llm->vocab_size) {
                        if (logits[tid] > 0.0f) {
                            logits[tid] /= conf.repetition_penalty;
                        } else {
                            logits[tid] *= conf.repetition_penalty;
                        }
                    }
                }
            }
        }

        int next_token = eif_llm_sample(llm, conf.temperature, conf.top_p);

        if (total_history < 4096) {
            history_tokens[total_history++] = next_token;
        }

        if (next_token == conf.eos_token_id || next_token == llm->tokenizer.eos_token_id ||
            next_token == 248046 || next_token == 248044 || next_token == 2) {
            break;
        }

        const char *piece = eif_bpe_tokenizer_decode(&llm->tokenizer, next_token);
        if (cb) {
            cb(piece, next_token, user_data);
        } else {
            /* Default stdout streaming */
            if (piece) {
                /* Strip BPE prefix symbol Ġ if present */
                if ((unsigned char)piece[0] == 0xC4 && (unsigned char)piece[1] == 0xA0) {
                    putchar(' ');
                    fputs(piece + 2, stdout);
                } else if ((unsigned char)piece[0] == 0xC4 && (unsigned char)piece[1] == 0x8A) {
                    putchar('\n');
                    fputs(piece + 2, stdout);
                } else {
                    fputs(piece, stdout);
                }
                fflush(stdout);
            }
        }

        generated_count++;
        if (eif_llm_forward(llm, next_token, pos++) != 0) {
            break;
        }
    }

    return generated_count;
}

void eif_llm_reset(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return;

    if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        qwen35_reset(&llm->backend.qwen35);
    } else {
        tinyllm_reset(&llm->backend.tinyllm);
    }
    llm->current_pos = 0;
}

void eif_llm_free(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return;

    if (llm->arch == EIF_LLM_ARCH_BERT) {
        eif_bert_free(&llm->backend.bert);
    } else if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        qwen35_free(&llm->backend.qwen35);
    } else {
        tinyllm_free(&llm->backend.tinyllm);
    }

    if (llm->tokenizer_loaded && llm->arch != EIF_LLM_ARCH_BERT) {
        eif_bpe_tokenizer_free(&llm->tokenizer);
        llm->tokenizer_loaded = false;
    }

    if (llm->owns_buffer && llm->buffer) {
        free(llm->buffer);
        llm->buffer = NULL;
    }

    eif_gpu_cleanup(&llm->gpu);

    llm->is_initialized = false;
}

float *eif_llm_get_hidden_state(eif_llm_t *llm)
{
    if (!llm || !llm->is_initialized) return NULL;
    if (llm->arch == EIF_LLM_ARCH_BERT) {
        return llm->backend.bert.scratch_seq_x;
    } else if (llm->arch == EIF_LLM_ARCH_QWEN35) {
        return qwen35_get_hidden_state(&llm->backend.qwen35);
    } else {
        return tinyllm_get_hidden_state(&llm->backend.tinyllm);
    }
}

int eif_llm_embed(eif_llm_t *llm, const char *prompt, float *out_embedding, eif_llm_pool_mode_t pool_mode)
{
    if (!llm || !llm->is_initialized || !prompt || !out_embedding) return -1;

    if (llm->arch == EIF_LLM_ARCH_BERT) {
        return eif_bert_embed(&llm->backend.bert, prompt, out_embedding);
    }

    if (!llm->tokenizer_loaded) return -2;

    int32_t prompt_tokens[2048];
    int n_prompt = eif_bpe_tokenizer_encode(&llm->tokenizer, prompt, 1, 0, prompt_tokens, 2048);
    if (n_prompt <= 0) return -3;

    int dim = llm->dim;
    memset(out_embedding, 0, (size_t)dim * sizeof(float));

    eif_llm_reset(llm);

    for (int i = 0; i < n_prompt; i++) {
        int token = prompt_tokens[i];
        if (eif_llm_forward_no_logits(llm, token, i) != 0) {
            return -4;
        }

        float *hidden = eif_llm_get_hidden_state(llm);
        if (!hidden) return -5;

        if (pool_mode == EIF_LLM_POOL_MEAN) {
            /* Accumulate for mean pooling */
            for (int d = 0; d < dim; d++) {
                out_embedding[d] += hidden[d];
            }
        } else if (pool_mode == EIF_LLM_POOL_LAST) {
            /* Keep last token */
            memcpy(out_embedding, hidden, (size_t)dim * sizeof(float));
        } else if (pool_mode == EIF_LLM_POOL_CLS && i == 0) {
            /* First token / CLS */
            memcpy(out_embedding, hidden, (size_t)dim * sizeof(float));
        }
    }

    if (pool_mode == EIF_LLM_POOL_MEAN && n_prompt > 1) {
        float inv_n = 1.0f / (float)n_prompt;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] *= inv_n;
        }
    }

    /* L2 Normalization */
    float sum_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        sum_sq += out_embedding[d] * out_embedding[d];
    }
    float norm = sqrtf(sum_sq);
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / norm;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] *= inv_norm;
        }
    }

    return 0;
}

float eif_llm_cosine_similarity(const float *a, const float *b, int dim)
{
    if (!a || !b || dim <= 0) return 0.0f;
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    return (denom > 1e-12f) ? (dot / denom) : 0.0f;
}
