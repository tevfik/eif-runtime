/**
 * @file eif_bert.c
 * @brief High-Performance Cross-Platform (x86_64 AVX2 / ARM64 NEON) BERT Runtime
 */

#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "eif_bert.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define EIF_ARCH_X86_64 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define EIF_ARCH_ARM64 1
#endif

/* Fast Half-Precision (FP16) to Single-Precision (FP32) Conversion */
static inline float fp16_to_fp32(uint16_t h)
{
#if defined(EIF_ARCH_X86_64)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#elif defined(EIF_ARCH_ARM64)
    __fp16 f16;
    memcpy(&f16, &h, 2);
    return (float)f16;
#else
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) return (sign) ? -0.0f : 0.0f;
        while (!(mant & 0x0400)) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp += 127 - 15;
    }
    uint32_t u = sign | (exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
#endif
}

/* =============================================================================
 * Cross-Platform SIMD Kernels
 * ============================================================================= */

/* 1. LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta */
static void bert_layernorm(float *out, const float *in, const float *gamma, const float *beta, int dim, float eps)
{
    float mean = 0.0f;
    float var = 0.0f;

#if defined(EIF_ARCH_X86_64)
    __m256 vmean = _mm256_setzero_ps();
    for (int i = 0; i <= dim - 8; i += 8) {
        vmean = _mm256_add_ps(vmean, _mm256_loadu_ps(&in[i]));
    }
    __m128 lo = _mm256_castps256_ps128(vmean);
    __m128 hi = _mm256_extractf128_ps(vmean, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    mean = _mm_cvtss_f32(s);
    for (int i = (dim & ~7); i < dim; i++) mean += in[i];
#elif defined(EIF_ARCH_ARM64)
    float32x4_t vmean = vdupq_n_f32(0.0f);
    for (int i = 0; i <= dim - 4; i += 4) {
        vmean = vaddq_f32(vmean, vld1q_f32(&in[i]));
    }
    mean = vaddvq_f32(vmean);
    for (int i = (dim & ~3); i < dim; i++) mean += in[i];
#else
    for (int i = 0; i < dim; i++) mean += in[i];
#endif
    mean /= (float)dim;

#if defined(EIF_ARCH_X86_64)
    __m256 vvm = _mm256_set1_ps(mean);
    __m256 vvar = _mm256_setzero_ps();
    for (int i = 0; i <= dim - 8; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(&in[i]), vvm);
        vvar = _mm256_fmadd_ps(diff, diff, vvar);
    }
    lo = _mm256_castps256_ps128(vvar);
    hi = _mm256_extractf128_ps(vvar, 1);
    s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    var = _mm_cvtss_f32(s);
    for (int i = (dim & ~7); i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
#elif defined(EIF_ARCH_ARM64)
    float32x4_t vvm = vdupq_n_f32(mean);
    float32x4_t vvar = vdupq_n_f32(0.0f);
    for (int i = 0; i <= dim - 4; i += 4) {
        float32x4_t diff = vsubq_f32(vld1q_f32(&in[i]), vvm);
        vvar = vfmaq_f32(vvar, diff, diff);
    }
    var = vaddvq_f32(vvar);
    for (int i = (dim & ~3); i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
#else
    for (int i = 0; i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
#endif
    var /= (float)dim;
    float inv_std = 1.0f / sqrtf(var + eps);

#if defined(EIF_ARCH_X86_64)
    __m256 vinv = _mm256_set1_ps(inv_std);
    for (int i = 0; i <= dim - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        __m256 g = _mm256_loadu_ps(&gamma[i]);
        __m256 b = beta ? _mm256_loadu_ps(&beta[i]) : _mm256_setzero_ps();
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x, vvm), vinv);
        _mm256_storeu_ps(&out[i], _mm256_fmadd_ps(norm, g, b));
    }
    for (int i = (dim & ~7); i < dim; i++) {
        out[i] = ((in[i] - mean) * inv_std) * gamma[i] + (beta ? beta[i] : 0.0f);
    }
#elif defined(EIF_ARCH_ARM64)
    float32x4_t vinv = vdupq_n_f32(inv_std);
    for (int i = 0; i <= dim - 4; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        float32x4_t g = vld1q_f32(&gamma[i]);
        float32x4_t b = beta ? vld1q_f32(&beta[i]) : vdupq_n_f32(0.0f);
        float32x4_t norm = vmulq_f32(vsubq_f32(x, vvm), vinv);
        vst1q_f32(&out[i], vfmaq_f32(b, norm, g));
    }
    for (int i = (dim & ~3); i < dim; i++) {
        out[i] = ((in[i] - mean) * inv_std) * gamma[i] + (beta ? beta[i] : 0.0f);
    }
#else
    for (int i = 0; i < dim; i++) {
        out[i] = ((in[i] - mean) * inv_std) * gamma[i] + (beta ? beta[i] : 0.0f);
    }
#endif
}

/* 2. Fast GELU Activation */
static inline float gelu_f32(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

static void bert_gelu(float *out, const float *in, int n)
{
#pragma omp parallel for schedule(static) if(n >= 1024)
    for (int i = 0; i < n; i++) {
        out[i] = gelu_f32(in[i]);
    }
}

/* 3. High-Performance Batched GEMM: Y[T, rows] = X[T, cols] * W[rows, cols]^T + bias */
static void bert_gemm_int8(
    float *Y,               /* Output: [T, rows] */
    const int8_t *W,        /* Weights: [rows, cols] */
    const float *W_scales,  /* Scales:  [rows, cols / 32] */
    const float *X,         /* Input:   [T, cols] */
    const float *bias,      /* Bias:    [rows] (optional) */
    int T,
    int rows,
    int cols)
{
    int blocks_per_row = cols / 32;

#pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 4096)
    for (int r = 0; r < rows; r++) {
        const int8_t *w_row = W + (size_t)r * cols;
        const float *scales_row = W_scales + (size_t)r * blocks_per_row;
        float b = bias ? bias[r] : 0.0f;

        float sums[512];
        for (int t = 0; t < T; t++) sums[t] = 0.0f;

        for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
            const int8_t *wb = w_row + b_idx * 32;
            float scale = scales_row[b_idx];

#if defined(EIF_ARCH_X86_64)
            /* Unpack 32 int8 weights once for all T tokens */
            __m128i r8_0 = _mm_loadl_epi64((const __m128i*)(wb));
            __m128i r8_1 = _mm_loadl_epi64((const __m128i*)(wb + 8));
            __m128i r8_2 = _mm_loadl_epi64((const __m128i*)(wb + 16));
            __m128i r8_3 = _mm_loadl_epi64((const __m128i*)(wb + 24));

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_0));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_1));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_2));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_3));

            for (int t = 0; t < T; t++) {
                const float *xb = X + t * cols + b_idx * 32;
                __m256 in0 = _mm256_loadu_ps(xb);
                __m256 in1 = _mm256_loadu_ps(xb + 8);
                __m256 in2 = _mm256_loadu_ps(xb + 16);
                __m256 in3 = _mm256_loadu_ps(xb + 24);

                __m256 acc = _mm256_fmadd_ps(in0, w0, _mm256_setzero_ps());
                acc = _mm256_fmadd_ps(in1, w1, acc);
                acc = _mm256_fmadd_ps(in2, w2, acc);
                acc = _mm256_fmadd_ps(in3, w3, acc);

                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 sm = _mm_add_ps(lo, hi);
                sm = _mm_hadd_ps(sm, sm);
                sm = _mm_hadd_ps(sm, sm);
                sums[t] += _mm_cvtss_f32(sm) * scale;
            }
#elif defined(EIF_ARCH_ARM64)
            float32x4_t wf[8];
            for (int k = 0; k < 8; k++) {
                int32_t val32;
                memcpy(&val32, wb + k * 4, 4);
                int8x8_t r8 = vreinterpret_s8_s32(vdup_n_s32(val32));
                int16x4_t r16 = vget_low_s16(vmovl_s8(r8));
                wf[k] = vcvtq_f32_s32(vmovl_s16(r16));
            }
            for (int t = 0; t < T; t++) {
                const float *xb = X + t * cols + b_idx * 32;
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int k = 0; k < 8; k++) {
                    acc = vfmaq_f32(acc, vld1q_f32(xb + k * 4), wf[k]);
                }
                sums[t] += vaddvq_f32(acc) * scale;
            }
#else
            for (int t = 0; t < T; t++) {
                const float *xb = X + t * cols + b_idx * 32;
                float block_sum = 0.0f;
                for (int c = 0; c < 32; c++) {
                    block_sum += (float)wb[c] * xb[c];
                }
                sums[t] += block_sum * scale;
            }
#endif
        }

        for (int t = 0; t < T; t++) {
            Y[t * rows + r] = sums[t] + b;
        }
    }
}

static void bert_gemm_f32(
    float *Y,
    const float *W,
    const float *X,
    const float *bias,
    int T,
    int rows,
    int cols)
{
#pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 4096)
    for (int r = 0; r < rows; r++) {
        const float *w_row = W + (size_t)r * cols;
        float b = bias ? bias[r] : 0.0f;
        float sums[512];
        for (int t = 0; t < T; t++) sums[t] = 0.0f;

        for (int c = 0; c <= cols - 8; c += 8) {
#if defined(EIF_ARCH_X86_64)
            __m256 wv = _mm256_loadu_ps(&w_row[c]);
            for (int t = 0; t < T; t++) {
                __m256 in_v = _mm256_loadu_ps(&X[t * cols + c]);
                __m256 prod = _mm256_mul_ps(in_v, wv);
                __m128 lo = _mm256_castps256_ps128(prod);
                __m128 hi = _mm256_extractf128_ps(prod, 1);
                __m128 sm = _mm_add_ps(lo, hi);
                sm = _mm_hadd_ps(sm, sm);
                sm = _mm_hadd_ps(sm, sm);
                sums[t] += _mm_cvtss_f32(sm);
            }
#elif defined(EIF_ARCH_ARM64)
            float32x4_t w0 = vld1q_f32(&w_row[c]);
            float32x4_t w1 = vld1q_f32(&w_row[c + 4]);
            for (int t = 0; t < T; t++) {
                float32x4_t in0 = vld1q_f32(&X[t * cols + c]);
                float32x4_t in1 = vld1q_f32(&X[t * cols + c + 4]);
                float32x4_t p0 = vmulq_f32(in0, w0);
                float32x4_t p1 = vfmaq_f32(p0, in1, w1);
                sums[t] += vaddvq_f32(p1);
            }
#else
            for (int t = 0; t < T; t++) {
                for (int k = 0; k < 8; k++) {
                    sums[t] += w_row[c + k] * X[t * cols + c + k];
                }
            }
#endif
        }
        for (int c = (cols & ~7); c < cols; c++) {
            for (int t = 0; t < T; t++) {
                sums[t] += w_row[c] * X[t * cols + c];
            }
        }

        for (int t = 0; t < T; t++) {
            Y[t * rows + r] = sums[t] + b;
        }
    }
}

/* Native GGUF Q8_0 Interleaved (2B FP16 scale + 32B INT8 weights per 34-byte block) */
static void bert_gemm_int8_gguf(
    float *Y,               /* Output: [T, rows] */
    const uint8_t *W,       /* Interleaved Q8_0: [rows, (cols / 32) * 34] */
    const float *X,         /* Input:   [T, cols] */
    const float *bias,      /* Bias:    [rows] (optional) */
    int T,
    int rows,
    int cols)
{
    int blocks_per_row = cols / 32;
    size_t row_stride_bytes = (size_t)blocks_per_row * 34;

#pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 4096)
    for (int r = 0; r < rows; r++) {
        const uint8_t *w_row = W + (size_t)r * row_stride_bytes;
        float b = bias ? bias[r] : 0.0f;

        float sums[512];
        for (int t = 0; t < T; t++) sums[t] = 0.0f;

        for (int b_idx = 0; b_idx < blocks_per_row; b_idx++) {
            const uint8_t *blk = w_row + b_idx * 34;
            float scale = fp16_to_fp32(*(const uint16_t *)blk);
            const int8_t *wb = (const int8_t *)(blk + 2);

#if defined(EIF_ARCH_X86_64)
            /* Unpack 32 int8 weights once for all T tokens */
            __m128i r8_0 = _mm_loadl_epi64((const __m128i*)(wb));
            __m128i r8_1 = _mm_loadl_epi64((const __m128i*)(wb + 8));
            __m128i r8_2 = _mm_loadl_epi64((const __m128i*)(wb + 16));
            __m128i r8_3 = _mm_loadl_epi64((const __m128i*)(wb + 24));

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_0));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_1));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_2));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8_3));

            for (int t = 0; t < T; t++) {
                const float *xb = X + t * cols + b_idx * 32;
                __m256 in0 = _mm256_loadu_ps(xb);
                __m256 in1 = _mm256_loadu_ps(xb + 8);
                __m256 in2 = _mm256_loadu_ps(xb + 16);
                __m256 in3 = _mm256_loadu_ps(xb + 24);

                __m256 acc = _mm256_fmadd_ps(in0, w0, _mm256_setzero_ps());
                acc = _mm256_fmadd_ps(in1, w1, acc);
                acc = _mm256_fmadd_ps(in2, w2, acc);
                acc = _mm256_fmadd_ps(in3, w3, acc);

                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 sm = _mm_add_ps(lo, hi);
                sm = _mm_hadd_ps(sm, sm);
                sm = _mm_hadd_ps(sm, sm);
                sums[t] += _mm_cvtss_f32(sm) * scale;
            }
#elif defined(EIF_ARCH_ARM64)
            float32x4_t wf[8];
            for (int k = 0; k < 8; k++) {
                int32_t val32;
                memcpy(&val32, wb + k * 4, 4);
                int8x8_t r8 = vreinterpret_s8_s32(vdup_n_s32(val32));
                int16x4_t r16 = vget_low_s16(vmovl_s8(r8));
                wf[k] = vcvtq_f32_s32(vmovl_s16(r16));
            }
            for (int t = 0; t < T; t++) {
                const float *xb = X + t * cols + b_idx * 32;
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int k = 0; k < 8; k++) {
                    acc = vfmaq_f32(acc, vld1q_f32(xb + k * 4), wf[k]);
                }
                sums[t] += vaddvq_f32(acc) * scale;
            }
#else
            for (int t = 0; t < T; t++) {
                const float *xb = X + t * cols + b_idx * 32;
                float block_sum = 0.0f;
                for (int c = 0; c < 32; c++) {
                    block_sum += (float)wb[c] * xb[c];
                }
                sums[t] += block_sum * scale;
            }
#endif
        }

        for (int t = 0; t < T; t++) {
            Y[t * rows + r] = sums[t] + b;
        }
    }
}

/* Dispatcher for Batched GEMM */
static inline void bert_gemm(
    float *Y,
    const void *W,
    const float *W_scales,
    const float *X,
    const float *bias,
    int T,
    int rows,
    int cols,
    int qtype)
{
    if (qtype == 2) {
        bert_gemm_int8_gguf(Y, (const uint8_t *)W, X, bias, T, rows, cols);
    } else if (qtype == 1) {
        bert_gemm_int8(Y, (const int8_t *)W, W_scales, X, bias, T, rows, cols);
    } else {
        bert_gemm_f32(Y, (const float *)W, X, bias, T, rows, cols);
    }
}

/* =============================================================================
 * WordPiece / SentencePiece Subword Tokenizer
 * ============================================================================= */

static inline uint32_t hash_str(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h = (h ^ (uint8_t)*s++) * 16777619u;
    }
    return h;
}

#define EBERT_HASH_MASK 0x1FFFF /* 131,072 slots */

static int find_token_id(const eif_bert_vocab_t *vocab, const char *token_str)
{
    if (!vocab || !vocab->tokens) return -1;
    if (vocab->hash_table) {
        uint32_t idx = hash_str(token_str) & EBERT_HASH_MASK;
        for (int step = 0; step < 256; step++) {
            int32_t tid = vocab->hash_table[(idx + step) & EBERT_HASH_MASK];
            if (tid == -1) return -1;
            if (strcmp(vocab->tokens[tid], token_str) == 0) return (int)tid;
        }
        return -1;
    }
    for (int i = 0; i < vocab->vocab_size; i++) {
        if (vocab->tokens[i] && strcmp(vocab->tokens[i], token_str) == 0) {
            return i;
        }
    }
    return -1;
}

int eif_bert_tokenize(const eif_bert_t *bert, const char *text, int32_t *out_tokens, int max_tokens)
{
    if (!bert || !text || !out_tokens || max_tokens < 3) return -1;
    const eif_bert_vocab_t *v = &bert->vocab;

    int n_out = 0;
    out_tokens[n_out++] = (v->cls_id >= 0) ? v->cls_id : 101; /* [CLS] */

    char word[256];
    const char *p = text;

    while (*p && n_out < max_tokens - 1) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Punctuation handling: isolate ASCII punctuation */
        if (ispunct((unsigned char)*p)) {
            char pchar = *p++;
            char punct_str[8];
            int tid = -1;
            if (v->is_spm_prefix) {
                punct_str[0] = (char)0xE2;
                punct_str[1] = (char)0x96;
                punct_str[2] = (char)0x81;
                punct_str[3] = pchar;
                punct_str[4] = '\0';
                tid = find_token_id(v, punct_str);
            }
            if (tid < 0) {
                punct_str[0] = pchar;
                punct_str[1] = '\0';
                tid = find_token_id(v, punct_str);
            }
            if (tid >= 0) {
                out_tokens[n_out++] = tid;
            }
            continue;
        }

        /* Read word until space or punctuation */
        int wlen = 0;
        while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && wlen < 200) {
            word[wlen++] = (char)tolower((unsigned char)*p++);
        }
        word[wlen] = '\0';
        if (wlen == 0) continue;

        /* Subword matching */
        int start = 0;
        while (start < wlen && n_out < max_tokens - 1) {
            int end = wlen;
            int found_id = -1;
            while (end > start) {
                char sub[256];
                int sub_len = end - start;
                if (start == 0) {
                    if (v->is_spm_prefix) {
                        sub[0] = (char)0xE2;
                        sub[1] = (char)0x96;
                        sub[2] = (char)0x81;
                        memcpy(sub + 3, word + start, sub_len);
                        sub[sub_len + 3] = '\0';
                    } else {
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                    }
                } else {
                    if (v->is_spm_prefix) {
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                    } else {
                        sub[0] = '#';
                        sub[1] = '#';
                        memcpy(sub + 2, word + start, sub_len);
                        sub[sub_len + 2] = '\0';
                    }
                }
                found_id = find_token_id(v, sub);
                if (found_id >= 0) {
                    out_tokens[n_out++] = found_id;
                    start = end;
                    break;
                }
                end--;
            }
            if (found_id < 0) {
                /* If start == 0 and spm prefix didn't match, fallback without spm prefix */
                if (start == 0 && v->is_spm_prefix) {
                    end = wlen;
                    while (end > start) {
                        char sub[256];
                        int sub_len = end - start;
                        memcpy(sub, word + start, sub_len);
                        sub[sub_len] = '\0';
                        found_id = find_token_id(v, sub);
                        if (found_id >= 0) {
                            out_tokens[n_out++] = found_id;
                            start = end;
                            break;
                        }
                        end--;
                    }
                }
                if (found_id < 0) {
                    out_tokens[n_out++] = (v->unk_id >= 0) ? v->unk_id : 100;
                    break;
                }
            }
        }
    }

    if (n_out < max_tokens) {
        out_tokens[n_out++] = (v->sep_id >= 0) ? v->sep_id : 102; /* [SEP] */
    }

    return n_out;
}

/* =============================================================================
 * Model Loading (Zero-Copy mmap)
 * ============================================================================= */

static inline size_t align64(size_t sz)
{
    return (sz + 63) & ~63;
}

/* Helpers for GGUF binary reading */
typedef struct {
    char name[128];
    uint32_t type;
    uint64_t offset;
} gguf_tensor_desc_t;

static inline uint32_t gguf_read_u32(const uint8_t **p) {
    uint32_t v; memcpy(&v, *p, 4); *p += 4; return v;
}
static inline uint64_t gguf_read_u64(const uint8_t **p) {
    uint64_t v; memcpy(&v, *p, 8); *p += 8; return v;
}
static inline void gguf_read_str(const uint8_t **p, char *out, size_t max_out) {
    uint64_t len = gguf_read_u64(p);
    size_t copy_len = (len < max_out - 1) ? len : (max_out - 1);
    memcpy(out, *p, copy_len);
    out[copy_len] = '\0';
    *p += len;
}

static const void *find_gguf_tensor(
    const gguf_tensor_desc_t *tensors,
    int n_tensors,
    const uint8_t *data_start,
    const char *name,
    uint32_t *out_type)
{
    for (int i = 0; i < n_tensors; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            if (out_type) *out_type = tensors[i].type;
            return data_start + tensors[i].offset;
        }
    }
    return NULL;
}

/* Common setup for vocab hash table and scratch memory */
static int bert_setup_vocab_and_scratch(eif_bert_t *bert)
{
    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int max_t = bert->config.max_seq_len;

    /* Detect if vocabulary uses SentencePiece lower one-eighth block space prefix (\xe2\x96\x81) */
    bert->vocab.is_spm_prefix = false;
    for (int t = 0; t < bert->vocab.vocab_size && t < 2000; t++) {
        if (bert->vocab.tokens[t] &&
            (uint8_t)bert->vocab.tokens[t][0] == 0xE2 &&
            (uint8_t)bert->vocab.tokens[t][1] == 0x96 &&
            (uint8_t)bert->vocab.tokens[t][2] == 0x81) {
            bert->vocab.is_spm_prefix = true;
            break;
        }
    }

    /* Build fast O(1) hash lookup table (131,072 slots) */
    bert->vocab.hash_table = (int32_t *)malloc(131072 * sizeof(int32_t));
    if (!bert->vocab.hash_table) return -7;
    for (int i = 0; i < 131072; i++) bert->vocab.hash_table[i] = -1;
    for (int t = 0; t < bert->vocab.vocab_size; t++) {
        if (!bert->vocab.tokens[t]) continue;
        uint32_t idx = hash_str(bert->vocab.tokens[t]) & EBERT_HASH_MASK;
        for (int step = 0; step < 256; step++) {
            uint32_t slot = (idx + step) & EBERT_HASH_MASK;
            if (bert->vocab.hash_table[slot] == -1) {
                bert->vocab.hash_table[slot] = (int32_t)t;
                break;
            }
        }
    }

    /* Allocate Scratch Activation Memory */
    bert->scratch_seq_x  = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_seq_xb = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_q      = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_k      = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_v      = (float *)malloc((size_t)max_t * dim * sizeof(float));
    bert->scratch_att    = (float *)malloc((size_t)bert->config.n_heads * max_t * max_t * sizeof(float));
    bert->scratch_inter  = (float *)malloc((size_t)max_t * inter_dim * sizeof(float));
    bert->scratch_proj   = (float *)malloc((size_t)max_t * dim * sizeof(float));

    if (!bert->scratch_seq_x || !bert->scratch_seq_xb || !bert->scratch_q ||
        !bert->scratch_k || !bert->scratch_v || !bert->scratch_att ||
        !bert->scratch_inter || !bert->scratch_proj) {
        return -8;
    }

    bert->is_initialized = true;
    return 0;
}

/* 1. Native GGUF BERT Loader */
static int eif_bert_load_gguf(eif_bert_t *bert, const uint8_t *addr, size_t file_sz)
{
    const uint8_t *p = addr;
    uint32_t magic = gguf_read_u32(&p);
    uint32_t ver   = gguf_read_u32(&p);
    uint64_t n_tensors = gguf_read_u64(&p);
    uint64_t n_kv      = gguf_read_u64(&p);

    if (magic != GGUF_MAGIC) return -5;
    (void)ver;

    uint32_t alignment = 32;
    int dim = 384, inter_dim = 1536, n_layers = 6, n_heads = 12, max_seq = 512;

    for (uint64_t i = 0; i < n_kv; i++) {
        char key[128];
        gguf_read_str(&p, key, sizeof(key));
        uint32_t vtype = gguf_read_u32(&p);

        if (strcmp(key, "general.alignment") == 0) {
            alignment = gguf_read_u32(&p);
        } else if (strcmp(key, "bert.embedding_length") == 0) {
            dim = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "bert.feed_forward_length") == 0) {
            inter_dim = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "bert.block_count") == 0) {
            n_layers = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "bert.attention.head_count") == 0) {
            n_heads = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "bert.context_length") == 0) {
            max_seq = (int)gguf_read_u32(&p);
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            uint32_t itype = gguf_read_u32(&p);
            uint64_t count = gguf_read_u64(&p);
            (void)itype;

            bert->vocab.vocab_size = (int)count;
            bert->vocab.tokens = (char **)malloc((size_t)count * sizeof(char *));
            bert->vocab.cls_id = 101;
            bert->vocab.sep_id = 102;
            bert->vocab.unk_id = 100;
            bert->vocab.pad_id = 0;

            for (uint64_t t = 0; t < count; t++) {
                uint64_t slen = gguf_read_u64(&p);
                bert->vocab.tokens[t] = (char *)malloc(slen + 1);
                memcpy(bert->vocab.tokens[t], p, slen);
                bert->vocab.tokens[t][slen] = '\0';
                p += slen;

                if (strcmp(bert->vocab.tokens[t], "[CLS]") == 0) bert->vocab.cls_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[SEP]") == 0) bert->vocab.sep_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[UNK]") == 0) bert->vocab.unk_id = (int)t;
                else if (strcmp(bert->vocab.tokens[t], "[PAD]") == 0) bert->vocab.pad_id = (int)t;
            }
        } else {
            /* Skip metadata value */
            switch (vtype) {
                case 0: case 1: case 7: p += 1; break;
                case 2: case 3: p += 2; break;
                case 4: case 5: case 6: p += 4; break;
                case 8: { uint64_t l = gguf_read_u64(&p); p += l; break; }
                case 10: case 11: case 12: p += 8; break;
                case 9: {
                    uint32_t atype = gguf_read_u32(&p);
                    uint64_t acount = gguf_read_u64(&p);
                    if (atype == 8) {
                        for (uint64_t k = 0; k < acount; k++) { uint64_t sl = gguf_read_u64(&p); p += sl; }
                    } else if (atype == 0 || atype == 1 || atype == 7) p += acount;
                    else if (atype == 2 || atype == 3) p += acount * 2;
                    else if (atype == 4 || atype == 5 || atype == 6) p += acount * 4;
                    else if (atype == 10 || atype == 11 || atype == 12) p += acount * 8;
                    break;
                }
                default:
                    return -6;
            }
        }
    }

    bert->config.dim = dim;
    bert->config.intermediate_dim = inter_dim;
    bert->config.n_layers = n_layers;
    bert->config.n_heads = n_heads;
    bert->config.max_seq_len = max_seq;

    /* Parse tensor directory */
    gguf_tensor_desc_t *tdescs = (gguf_tensor_desc_t *)malloc((size_t)n_tensors * sizeof(gguf_tensor_desc_t));
    if (!tdescs) return -7;

    for (uint64_t t = 0; t < n_tensors; t++) {
        gguf_read_str(&p, tdescs[t].name, sizeof(tdescs[t].name));
        uint32_t ndims = gguf_read_u32(&p);
        for (uint32_t d = 0; d < ndims; d++) gguf_read_u64(&p);
        tdescs[t].type = gguf_read_u32(&p);
        tdescs[t].offset = gguf_read_u64(&p);
    }

    size_t header_len = (size_t)(p - addr);
    size_t data_start_offset = (header_len + alignment - 1) & ~(size_t)(alignment - 1);
    const uint8_t *data_start = addr + data_start_offset;

    uint32_t tok_type = 0;
    bert->weights.token_emb = find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd.weight", &tok_type);
    bert->config.qtype = (tok_type == 8) ? 2 : 0; /* 2 = Native GGUF Q8_0 interleaved */

    bert->weights.pos_emb    = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "position_embd.weight", NULL);
    bert->weights.type_emb   = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_types.weight", NULL);
    bert->weights.emb_norm_w = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd_norm.weight", NULL);
    bert->weights.emb_norm_b = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, "token_embd_norm.bias", NULL);

    size_t arr_sz = (size_t)n_layers * sizeof(void *);
    bert->weights.q_w            = (const void **)malloc(arr_sz);
    bert->weights.q_w_scales     = NULL;
    bert->weights.q_b            = (const float **)malloc(arr_sz);
    bert->weights.k_w            = (const void **)malloc(arr_sz);
    bert->weights.k_w_scales     = NULL;
    bert->weights.k_b            = (const float **)malloc(arr_sz);
    bert->weights.v_w            = (const void **)malloc(arr_sz);
    bert->weights.v_w_scales     = NULL;
    bert->weights.v_b            = (const float **)malloc(arr_sz);
    bert->weights.out_w          = (const void **)malloc(arr_sz);
    bert->weights.out_w_scales   = NULL;
    bert->weights.out_b          = (const float **)malloc(arr_sz);
    bert->weights.att_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.att_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_w       = (const void **)malloc(arr_sz);
    bert->weights.ffn_up_w_scales = NULL;
    bert->weights.ffn_up_b       = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_w     = (const void **)malloc(arr_sz);
    bert->weights.ffn_down_w_scales = NULL;
    bert->weights.ffn_down_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_b     = (const float **)malloc(arr_sz);

    char tname[128];
    for (int l = 0; l < n_layers; l++) {
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        bert->weights.q_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.bias", l);
        bert->weights.q_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        bert->weights.k_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_k.bias", l);
        bert->weights.k_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
        bert->weights.v_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_v.bias", l);
        bert->weights.v_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
        bert->weights.out_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_output.bias", l);
        bert->weights.out_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.attn_output_norm.weight", l);
        bert->weights.att_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.attn_output_norm.bias", l);
        bert->weights.att_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
        bert->weights.ffn_up_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.bias", l);
        bert->weights.ffn_up_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
        bert->weights.ffn_down_w[l] = find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.bias", l);
        bert->weights.ffn_down_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);

        snprintf(tname, sizeof(tname), "blk.%d.layer_output_norm.weight", l);
        bert->weights.ffn_norm_w[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
        snprintf(tname, sizeof(tname), "blk.%d.layer_output_norm.bias", l);
        bert->weights.ffn_norm_b[l] = (const float *)find_gguf_tensor(tdescs, (int)n_tensors, data_start, tname, NULL);
    }
    free(tdescs);

    return bert_setup_vocab_and_scratch(bert);
}

/* 2. EIFM Binary Loader */
static int eif_bert_load_eifm(eif_bert_t *bert, const uint8_t *addr, size_t file_sz)
{
    const uint32_t *h32 = (const uint32_t *)addr;
    if (h32[0] != EBERT_MAGIC) return -5;

    bert->config.dim              = (int)h32[2];
    bert->config.intermediate_dim = (int)h32[3];
    bert->config.n_layers         = (int)h32[4];
    bert->config.n_heads          = (int)h32[5];
    bert->config.max_seq_len      = (int)h32[6];
    bert->config.vocab_size       = (int)h32[7];
    bert->config.qtype            = (int)h32[8];

    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int n_layers = bert->config.n_layers;
    int vocab_size = bert->config.vocab_size;
    int qtype = bert->config.qtype;
    int blocks_dim = dim / 32;
    int blocks_inter = inter_dim / 32;

    const uint8_t *ptr = addr + 64;
    #define ADVANCE_PTR(p, bytes) do { p += align64(bytes); } while(0)

    bert->weights.token_emb = ptr;
    size_t token_emb_bytes = (qtype == 1) ? ((size_t)vocab_size * dim) : ((size_t)vocab_size * dim * sizeof(float));
    ADVANCE_PTR(ptr, token_emb_bytes);

    if (qtype == 1) {
        bert->weights.token_emb_scales = (const float *)ptr;
        ADVANCE_PTR(ptr, (size_t)vocab_size * blocks_dim * sizeof(float));
    }

    bert->weights.pos_emb = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)bert->config.max_seq_len * dim * sizeof(float));

    bert->weights.type_emb = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)2 * dim * sizeof(float));

    bert->weights.emb_norm_w = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

    bert->weights.emb_norm_b = (const float *)ptr;
    ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

    size_t arr_sz = (size_t)n_layers * sizeof(void *);
    bert->weights.q_w            = (const void **)malloc(arr_sz);
    bert->weights.q_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.q_b            = (const float **)malloc(arr_sz);
    bert->weights.k_w            = (const void **)malloc(arr_sz);
    bert->weights.k_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.k_b            = (const float **)malloc(arr_sz);
    bert->weights.v_w            = (const void **)malloc(arr_sz);
    bert->weights.v_w_scales     = (const float **)malloc(arr_sz);
    bert->weights.v_b            = (const float **)malloc(arr_sz);
    bert->weights.out_w          = (const void **)malloc(arr_sz);
    bert->weights.out_w_scales   = (const float **)malloc(arr_sz);
    bert->weights.out_b          = (const float **)malloc(arr_sz);
    bert->weights.att_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.att_norm_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_w       = (const void **)malloc(arr_sz);
    bert->weights.ffn_up_w_scales = (const float **)malloc(arr_sz);
    bert->weights.ffn_up_b       = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_w     = (const void **)malloc(arr_sz);
    bert->weights.ffn_down_w_scales = (const float **)malloc(arr_sz);
    bert->weights.ffn_down_b     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_w     = (const float **)malloc(arr_sz);
    bert->weights.ffn_norm_b     = (const float **)malloc(arr_sz);

    size_t lin_dim_bytes = (qtype == 1) ? ((size_t)dim * dim) : ((size_t)dim * dim * sizeof(float));
    size_t lin_dim_scales_bytes = (size_t)dim * blocks_dim * sizeof(float);
    size_t up_bytes = (qtype == 1) ? ((size_t)inter_dim * dim) : ((size_t)inter_dim * dim * sizeof(float));
    size_t up_scales_bytes = (size_t)inter_dim * blocks_dim * sizeof(float);
    size_t down_bytes = (qtype == 1) ? ((size_t)dim * inter_dim) : ((size_t)dim * inter_dim * sizeof(float));
    size_t down_scales_bytes = (size_t)dim * blocks_inter * sizeof(float);

    for (int l = 0; l < n_layers; l++) {
        bert->weights.q_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.q_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.q_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.k_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.k_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.k_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.v_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.v_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.v_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.out_w[l] = ptr; ADVANCE_PTR(ptr, lin_dim_bytes);
        if (qtype == 1) { bert->weights.out_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, lin_dim_scales_bytes); }
        bert->weights.out_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.att_norm_w[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
        bert->weights.att_norm_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.ffn_up_w[l] = ptr; ADVANCE_PTR(ptr, up_bytes);
        if (qtype == 1) { bert->weights.ffn_up_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, up_scales_bytes); }
        bert->weights.ffn_up_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)inter_dim * sizeof(float));

        bert->weights.ffn_down_w[l] = ptr; ADVANCE_PTR(ptr, down_bytes);
        if (qtype == 1) { bert->weights.ffn_down_w_scales[l] = (const float *)ptr; ADVANCE_PTR(ptr, down_scales_bytes); }
        bert->weights.ffn_down_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));

        bert->weights.ffn_norm_w[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
        bert->weights.ffn_norm_b[l] = (const float *)ptr; ADVANCE_PTR(ptr, (size_t)dim * sizeof(float));
    }

    /* Vocabulary Table */
    if (ptr < addr + file_sz) {
        uint32_t n_tok = *(const uint32_t *)ptr;
        ptr += 4;
        bert->vocab.vocab_size = (int)n_tok;
        bert->vocab.tokens = (char **)malloc((size_t)n_tok * sizeof(char *));
        bert->vocab.cls_id = 101;
        bert->vocab.sep_id = 102;
        bert->vocab.unk_id = 100;
        bert->vocab.pad_id = 0;

        for (uint32_t t = 0; t < n_tok && ptr < addr + file_sz; t++) {
            uint16_t tlen = *(const uint16_t *)ptr;
            ptr += 2;
            bert->vocab.tokens[t] = (char *)malloc(tlen + 1);
            memcpy(bert->vocab.tokens[t], ptr, tlen);
            bert->vocab.tokens[t][tlen] = '\0';
            ptr += tlen;

            if (strcmp(bert->vocab.tokens[t], "[CLS]") == 0) bert->vocab.cls_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[SEP]") == 0) bert->vocab.sep_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[UNK]") == 0) bert->vocab.unk_id = (int)t;
            else if (strcmp(bert->vocab.tokens[t], "[PAD]") == 0) bert->vocab.pad_id = (int)t;
        }
    }

    return bert_setup_vocab_and_scratch(bert);
}

/* 3. Polymorphic BERT Loader (GGUF or EIFM) */
int eif_bert_load(eif_bert_t *bert, const char *model_path)
{
    if (!bert || !model_path) return -1;
    memset(bert, 0, sizeof(eif_bert_t));

#if defined(_WIN32)
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "eif_bert_load: failed to open '%s'\n", model_path);
        return -2;
    }

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        CloseHandle(hFile);
        return -3;
    }
    size_t file_sz = (size_t)liSize.QuadPart;
    if (file_sz < 64) {
        CloseHandle(hFile);
        return -4;
    }

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        return -4;
    }

    void *addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, file_sz);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!addr) {
        fprintf(stderr, "eif_bert_load MapViewOfFile failed for '%s'\n", model_path);
        return -4;
    }
#else
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        perror("eif_bert_load open");
        return -2;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -3;
    }
    size_t file_sz = (size_t)st.st_size;
    if (file_sz < 64) {
        close(fd);
        return -4;
    }

    void *addr = mmap(NULL, file_sz, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        perror("eif_bert_load mmap");
        return -4;
    }
#endif

    bert->mmap_addr = addr;
    bert->mmap_size = file_sz;
    bert->is_mmap = true;

    uint32_t magic = *(const uint32_t *)addr;
    if (magic == GGUF_MAGIC) {
        int rc = eif_bert_load_gguf(bert, (const uint8_t *)addr, file_sz);
        if (rc != 0) {
            eif_bert_free(bert);
            return rc;
        }
        return 0;
    } else if (magic == EBERT_MAGIC) {
        int rc = eif_bert_load_eifm(bert, (const uint8_t *)addr, file_sz);
        if (rc != 0) {
            eif_bert_free(bert);
            return rc;
        }
        return 0;
    } else {
        fprintf(stderr, "eif_bert_load: Unsupported model magic 0x%08X in '%s' (expected GGUF or BERT)\n", magic, model_path);
        eif_bert_free(bert);
        return -5;
    }
}

void eif_bert_free(eif_bert_t *bert)
{
    if (!bert || !bert->is_initialized) return;

    if (bert->weights.q_w) free((void *)bert->weights.q_w);
    if (bert->weights.q_w_scales) free((void *)bert->weights.q_w_scales);
    if (bert->weights.q_b) free((void *)bert->weights.q_b);
    if (bert->weights.k_w) free((void *)bert->weights.k_w);
    if (bert->weights.k_w_scales) free((void *)bert->weights.k_w_scales);
    if (bert->weights.k_b) free((void *)bert->weights.k_b);
    if (bert->weights.v_w) free((void *)bert->weights.v_w);
    if (bert->weights.v_w_scales) free((void *)bert->weights.v_w_scales);
    if (bert->weights.v_b) free((void *)bert->weights.v_b);
    if (bert->weights.out_w) free((void *)bert->weights.out_w);
    if (bert->weights.out_w_scales) free((void *)bert->weights.out_w_scales);
    if (bert->weights.out_b) free((void *)bert->weights.out_b);
    if (bert->weights.att_norm_w) free((void *)bert->weights.att_norm_w);
    if (bert->weights.att_norm_b) free((void *)bert->weights.att_norm_b);
    if (bert->weights.ffn_up_w) free((void *)bert->weights.ffn_up_w);
    if (bert->weights.ffn_up_w_scales) free((void *)bert->weights.ffn_up_w_scales);
    if (bert->weights.ffn_up_b) free((void *)bert->weights.ffn_up_b);
    if (bert->weights.ffn_down_w) free((void *)bert->weights.ffn_down_w);
    if (bert->weights.ffn_down_w_scales) free((void *)bert->weights.ffn_down_w_scales);
    if (bert->weights.ffn_down_b) free((void *)bert->weights.ffn_down_b);
    if (bert->weights.ffn_norm_w) free((void *)bert->weights.ffn_norm_w);
    if (bert->weights.ffn_norm_b) free((void *)bert->weights.ffn_norm_b);

    if (bert->vocab.tokens) {
        for (int i = 0; i < bert->vocab.vocab_size; i++) {
            if (bert->vocab.tokens[i]) free(bert->vocab.tokens[i]);
        }
        free(bert->vocab.tokens);
    }
    if (bert->vocab.hash_table) {
        free(bert->vocab.hash_table);
    }

    if (bert->scratch_seq_x) free(bert->scratch_seq_x);
    if (bert->scratch_seq_xb) free(bert->scratch_seq_xb);
    if (bert->scratch_q) free(bert->scratch_q);
    if (bert->scratch_k) free(bert->scratch_k);
    if (bert->scratch_v) free(bert->scratch_v);
    if (bert->scratch_att) free(bert->scratch_att);
    if (bert->scratch_inter) free(bert->scratch_inter);
    if (bert->scratch_proj) free(bert->scratch_proj);

    if (bert->is_mmap && bert->mmap_addr) {
#if defined(_WIN32)
        UnmapViewOfFile(bert->mmap_addr);
#else
        munmap(bert->mmap_addr, bert->mmap_size);
#endif
    }

    bert->is_initialized = false;
}

/* =============================================================================
 * Forward Pass (Bidirectional Multi-Head Attention + FFN)
 * ============================================================================= */

int eif_bert_embed(eif_bert_t *bert, const char *text, float *out_embedding)
{
    if (!bert || !bert->is_initialized || !text || !out_embedding) return -1;

    int max_tokens = bert->config.max_seq_len;
    int32_t tokens[512];
    int T = eif_bert_tokenize(bert, text, tokens, (max_tokens > 512) ? 512 : max_tokens);
    if (T <= 0) return -2;

    int dim = bert->config.dim;
    int inter_dim = bert->config.intermediate_dim;
    int n_layers = bert->config.n_layers;
    int n_heads = bert->config.n_heads;
    int head_dim = dim / n_heads;
    int qtype = bert->config.qtype;
    int blocks_dim = dim / 32;
    float inv_sqrt_head = 1.0f / sqrtf((float)head_dim);

    float *X = bert->scratch_seq_x;
    float *XB = bert->scratch_seq_xb;
    float *Q = bert->scratch_q;
    float *K = bert->scratch_k;
    float *V = bert->scratch_v;
    float *ATT = bert->scratch_att;
    float *H_INTER = bert->scratch_inter;
    float *PROJ = bert->scratch_proj;

    /* 1. Embedding Layer: Word + Pos + Token_Type -> LayerNorm */
    for (int t = 0; t < T; t++) {
        int tid = tokens[t];
        float *x_t = X + t * dim;

        if (qtype == 2) {
            /* Native GGUF Q8_0 interleaved (2B FP16 scale + 32B INT8 weights) */
            const uint8_t *w_row = (const uint8_t *)bert->weights.token_emb + (size_t)tid * (blocks_dim * 34);
            for (int b = 0; b < blocks_dim; b++) {
                const uint8_t *blk = w_row + b * 34;
                float scale = fp16_to_fp32(*(const uint16_t *)blk);
                const int8_t *wb = (const int8_t *)(blk + 2);
                for (int c = 0; c < 32; c++) {
                    x_t[b * 32 + c] = (float)wb[c] * scale;
                }
            }
        } else if (qtype == 1) {
            const int8_t *w_tok = (const int8_t *)bert->weights.token_emb + (size_t)tid * dim;
            const float *s_tok = bert->weights.token_emb_scales + (size_t)tid * blocks_dim;
            for (int b = 0; b < blocks_dim; b++) {
                float scale = s_tok[b];
                for (int c = 0; c < 32; c++) {
                    x_t[b * 32 + c] = (float)w_tok[b * 32 + c] * scale;
                }
            }
        } else {
            const float *w_tok = (const float *)bert->weights.token_emb + (size_t)tid * dim;
            memcpy(x_t, w_tok, dim * sizeof(float));
        }

        /* Add Position and Segment Embeddings */
        const float *pos_row = bert->weights.pos_emb + (size_t)t * dim;
        const float *type_row = bert->weights.type_emb; /* segment 0 */
        for (int d = 0; d < dim; d++) {
            x_t[d] += pos_row[d] + (type_row ? type_row[d] : 0.0f);
        }

        /* Embedding LayerNorm */
        bert_layernorm(x_t, x_t, bert->weights.emb_norm_w, bert->weights.emb_norm_b, dim, 1e-12f);
    }

    /* 2. Transformer Layers */
    for (int l = 0; l < n_layers; l++) {
        /* 2a. Batched Q, K, V Projections for all T tokens */
        bert_gemm(Q, bert->weights.q_w[l], bert->weights.q_w_scales ? bert->weights.q_w_scales[l] : NULL,
                  X, bert->weights.q_b[l], T, dim, dim, qtype);
        bert_gemm(K, bert->weights.k_w[l], bert->weights.k_w_scales ? bert->weights.k_w_scales[l] : NULL,
                  X, bert->weights.k_b[l], T, dim, dim, qtype);
        bert_gemm(V, bert->weights.v_w[l], bert->weights.v_w_scales ? bert->weights.v_w_scales[l] : NULL,
                  X, bert->weights.v_b[l], T, dim, dim, qtype);

        /* 2b. Multi-Head Bidirectional Self-Attention */
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < n_heads; h++) {
            float *att_h = ATT + h * T * T;
            for (int i = 0; i < T; i++) {
                const float *q_i = Q + i * dim + h * head_dim;
                float max_val = -1e9f;

                /* Dot-product Q_i and K_j across all j (no causal mask!) */
                for (int j = 0; j < T; j++) {
                    const float *k_j = K + j * dim + h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += q_i[d] * k_j[d];
                    }
                    score *= inv_sqrt_head;
                    att_h[i * T + j] = score;
                    if (score > max_val) max_val = score;
                }

                /* Softmax */
                float sum_exp = 0.0f;
                for (int j = 0; j < T; j++) {
                    float ex = expf(att_h[i * T + j] - max_val);
                    att_h[i * T + j] = ex;
                    sum_exp += ex;
                }
                float inv_sum = 1.0f / (sum_exp > 1e-12f ? sum_exp : 1.0f);
                for (int j = 0; j < T; j++) {
                    att_h[i * T + j] *= inv_sum;
                }
            }
        }

        /* 2c. Context Accumulation: C = Attn * V */
        memset(XB, 0, (size_t)T * dim * sizeof(float));
        for (int h = 0; h < n_heads; h++) {
            const float *att_h = ATT + h * T * T;
            for (int i = 0; i < T; i++) {
                float *c_ih = XB + i * dim + h * head_dim;
                for (int j = 0; j < T; j++) {
                    float p_ij = att_h[i * T + j];
                    const float *v_jh = V + j * dim + h * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        c_ih[d] += p_ij * v_jh[d];
                    }
                }
            }
        }

        /* 2d. Output Projection: PROJ = XB * W_out + b */
        bert_gemm(PROJ, bert->weights.out_w[l], bert->weights.out_w_scales ? bert->weights.out_w_scales[l] : NULL,
                  XB, bert->weights.out_b[l], T, dim, dim, qtype);

        /* Residual Connection + LayerNorm */
        for (int t = 0; t < T; t++) {
            float *x_t = X + t * dim;
            const float *p_t = PROJ + t * dim;
            for (int d = 0; d < dim; d++) {
                x_t[d] += p_t[d];
            }
            bert_layernorm(x_t, x_t, bert->weights.att_norm_w[l], bert->weights.att_norm_b[l], dim, 1e-12f);
        }

        /* 2e. Feed-Forward Network: H_INTER = GELU(X * W_up + b) */
        bert_gemm(H_INTER, bert->weights.ffn_up_w[l], bert->weights.ffn_up_w_scales ? bert->weights.ffn_up_w_scales[l] : NULL,
                  X, bert->weights.ffn_up_b[l], T, inter_dim, dim, qtype);

        bert_gelu(H_INTER, H_INTER, T * inter_dim);

        /* FFN Down: PROJ = H_INTER * W_down + b */
        bert_gemm(PROJ, bert->weights.ffn_down_w[l], bert->weights.ffn_down_w_scales ? bert->weights.ffn_down_w_scales[l] : NULL,
                  H_INTER, bert->weights.ffn_down_b[l], T, dim, inter_dim, qtype);

        /* Residual Connection + LayerNorm */
        for (int t = 0; t < T; t++) {
            float *x_t = X + t * dim;
            const float *f_t = PROJ + t * dim;
            for (int d = 0; d < dim; d++) {
                x_t[d] += f_t[d];
            }
            bert_layernorm(x_t, x_t, bert->weights.ffn_norm_w[l], bert->weights.ffn_norm_b[l], dim, 1e-12f);
        }
    }

    /* 3. Mean Pooling: Average all T token vectors */
    memset(out_embedding, 0, (size_t)dim * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *x_t = X + t * dim;
        for (int d = 0; d < dim; d++) {
            out_embedding[d] += x_t[d];
        }
    }
    float inv_T = 1.0f / (float)T;
    for (int d = 0; d < dim; d++) {
        out_embedding[d] *= inv_T;
    }

    /* 4. L2 Normalization */
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
