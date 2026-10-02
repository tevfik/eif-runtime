/**
 * @file eif_tinyllm.c
 * @brief Tiny LLM Implementation
 *
 * Minimal transformer for microcontroller deployment.
 * Supports INT4/INT8 quantization for memory efficiency.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include "eif_tinyllm.h"
#include "eif_quantize_bitnet.h"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// Memory Management
// =============================================================================

size_t tinyllm_memory_size(const tinyllm_config_t *config)
{
    size_t size = 0;

    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;

    // Activation buffers
    size += config->dim * sizeof(float);        // x
    size += config->dim * sizeof(float);        // xb
    size += config->dim * sizeof(float);        // xb2
    size += config->dim * sizeof(float);        // q
    size += kv_dim * sizeof(float);             // k (kv_dim, not dim)
    size += kv_dim * sizeof(float);             // v (kv_dim, not dim)
    size += config->hidden_dim * sizeof(float); // hb
    size += config->hidden_dim * sizeof(float); // hb2
    size += config->vocab_size * sizeof(float); // logits

    // KV cache (per layer: seq_len * kv_dim * 2 for k and v)
    size += config->n_layers * config->seq_len * kv_dim * 2 * sizeof(float);

    // Attention scores
    size += config->n_heads * config->seq_len * sizeof(float);

    return size;
}

int tinyllm_init(tinyllm_t *llm, const tinyllm_config_t *config, void *memory_pool,
                 size_t pool_size)
{
    memset(llm, 0, sizeof(tinyllm_t));
    memcpy(&llm->config, config, sizeof(tinyllm_config_t));

    size_t required = tinyllm_memory_size(config);
    if (pool_size < required) {
        return -1;
    }

    llm->memory_pool = memory_pool;
    llm->memory_size = pool_size;

    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;

    float *ptr = (float *)memory_pool;

    llm->state.x = ptr;
    ptr += config->dim;
    llm->state.xb = ptr;
    ptr += config->dim;
    llm->state.xb2 = ptr;
    ptr += config->dim;
    llm->state.q = ptr;
    ptr += config->dim;
    llm->state.k = ptr;
    ptr += kv_dim;
    llm->state.v = ptr;
    ptr += kv_dim;
    llm->state.hb = ptr;
    ptr += config->hidden_dim;
    llm->state.hb2 = ptr;
    ptr += config->hidden_dim;
    llm->state.logits = ptr;
    ptr += config->vocab_size;

    llm->state.att = ptr;
    ptr += config->n_heads * config->seq_len;

    llm->state.key_cache = ptr;
    ptr += config->n_layers * config->seq_len * kv_dim;
    llm->state.value_cache = ptr;

    return 0;
}

void tinyllm_free(tinyllm_t *llm)
{
    // Memory pool is externally managed
    memset(llm, 0, sizeof(tinyllm_t));
}

void tinyllm_reset(tinyllm_t *llm)
{
    llm->state.pos = 0;

    int kv_dim = (llm->config.dim * llm->config.n_kv_heads) / llm->config.n_heads;

    memset(llm->state.key_cache, 0,
           llm->config.n_layers * llm->config.seq_len * kv_dim * sizeof(float));
    memset(llm->state.value_cache, 0,
           llm->config.n_layers * llm->config.seq_len * kv_dim * sizeof(float));
}

// =============================================================================
// Core Operations
// =============================================================================

#if defined(__AVX2__)
#include <immintrin.h>
static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps)
{
    __m256 ss_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        ss_vec = _mm256_fmadd_ps(vx, vx, ss_vec);
    }
    __m128 low = _mm256_castps256_ps128(ss_vec);
    __m128 high = _mm256_extractf128_ps(ss_vec, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float ss = _mm_cvtss_f32(sum128);
    for (; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / size + eps;
    ss = 1.0f / sqrtf(ss);

    __m256 vss = _mm256_set1_ps(ss);
    i = 0;
    if (weight) {
        for (; i <= size - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            __m256 vw = _mm256_loadu_ps(&weight[i]);
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_mul_ps(vx, vss), vw));
        }
        for (; i < size; i++) {
            out[i] = x[i] * ss * weight[i];
        }
    } else {
        for (; i <= size - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(vx, vss));
        }
        for (; i < size; i++) {
            out[i] = x[i] * ss;
        }
    }
}
#else
static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps)
{
    // Calculate RMS
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / size + eps;
    ss = 1.0f / sqrtf(ss);

    // Normalize and scale
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * (weight ? weight[i] : 1.0f);
    }
}
#endif

static void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val)
            max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void matmul(float *xout, const float *x, const float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // W is stored row-major as (d, n)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// =============================================================================
// Forward Pass
// =============================================================================

float *tinyllm_forward_embedding(tinyllm_t *llm, const float *embedding, int pos)
{
    tinyllm_config_t *p = &llm->config;
    tinyllm_weights_t *w = &llm->weights;
    tinyllm_state_t *s = &llm->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = dim / p->n_heads;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;

    if (embedding != s->x) {
        memcpy(s->x, embedding, dim * sizeof(float));
    }

    /* Precompute RoPE cos/sin for this position (once for all 30 layers!) */
    int half_head = head_dim / 2;
    float rope_cos[64];
    float rope_sin[64];
    float rope_theta = (p->vocab_size > 60000) ? 100000.0f : 10000.0f;
    for (int j = 0; j < half_head && j < 64; j++) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * j) / (float)head_dim);
        float val = (float)pos * freq;
        rope_cos[j] = cosf(val);
        rope_sin[j] = sinf(val);
    }
    float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);

    // Process each layer
    for (int l = 0; l < p->n_layers; l++) {
        // Attention RMSNorm
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim, 1e-5f);

        // QKV matmuls
        int loff = l * p->seq_len * kv_dim;

        float *k = s->key_cache + loff + pos * kv_dim;
        float *v = s->value_cache + loff + pos * kv_dim;

        // matmul(xout, x, w, n, d): W(d,n) @ x(n,) -> xout(d,)
        if (p->qtype == TINYLLM_QTYPE_INT8) {
            int8_t *w_wq = (int8_t *)w->wq;
            int8_t *w_wk = (int8_t *)w->wk;
            int8_t *w_wv = (int8_t *)w->wv;
            tinyllm_matmul_int8(s->q, w_wq + l * dim * dim, s->xb, dim, dim,
                                w->attn_scale[l * 4 + 0]);
            tinyllm_matmul_int8(k, w_wk + l * dim * kv_dim, s->xb, dim, kv_dim,
                                w->attn_scale[l * 4 + 1]);
            tinyllm_matmul_int8(v, w_wv + l * dim * kv_dim, s->xb, dim, kv_dim,
                                w->attn_scale[l * 4 + 2]);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_wq = (uint8_t *)w->wq;
            uint8_t *w_wk = (uint8_t *)w->wk;
            uint8_t *w_wv = (uint8_t *)w->wv;
            int dim_packed = (dim + 1) / 2;
            int kv_dim_packed = (dim + 1) / 2;
            tinyllm_matmul_int4(s->q, w_wq + l * dim * dim_packed, s->xb, dim, dim,
                                w->attn_scale + l * (dim * 4 + kv_dim * 2));
            tinyllm_matmul_int4(k, w_wk + l * kv_dim * kv_dim_packed, s->xb, dim, kv_dim,
                                w->attn_scale + l * (dim * 4 + kv_dim * 2) + dim);
            tinyllm_matmul_int4(v, w_wv + l * kv_dim * kv_dim_packed, s->xb, dim, kv_dim,
                                w->attn_scale + l * (dim * 4 + kv_dim * 2) + dim * 2);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            uint8_t *w_wq = (uint8_t *)w->wq;
            uint8_t *w_wk = (uint8_t *)w->wk;
            uint8_t *w_wv = (uint8_t *)w->wv;
            int dim_packed = (dim + 3) / 4;
            tinyllm_matmul_bitnet_158(s->q, w_wq + l * dim * dim_packed, s->xb, dim, dim,
                                     w->attn_scale + l * (dim * 4 + kv_dim * 2));
            tinyllm_matmul_bitnet_158(k, w_wk + l * kv_dim * dim_packed, s->xb, dim, kv_dim,
                                     w->attn_scale + l * (dim * 4 + kv_dim * 2) + dim);
            tinyllm_matmul_bitnet_158(v, w_wv + l * kv_dim * dim_packed, s->xb, dim, kv_dim,
                                     w->attn_scale + l * (dim * 4 + kv_dim * 2) + dim * 2);
        } else {
            float *w_wq = (float *)w->wq;
            float *w_wk = (float *)w->wk;
            float *w_wv = (float *)w->wv;
            matmul(s->q, s->xb, w_wq + l * dim * dim, dim, dim);
            matmul(k, s->xb, w_wk + l * dim * kv_dim, dim, kv_dim);
            matmul(v, s->xb, w_wv + l * dim * kv_dim, dim, kv_dim);
        }

        // RoPE positional encoding (HuggingFace LLaMA rotate_half standard)
        for (int h = 0; h < p->n_heads; h++) {
            float *qh = s->q + h * head_dim;
            for (int j = 0; j < half_head; j++) {
                float q0 = qh[j];
                float q1 = qh[j + half_head];
                float c = rope_cos[j];
                float s_val = rope_sin[j];
                qh[j]             = q0 * c - q1 * s_val;
                qh[j + half_head] = q1 * c + q0 * s_val;
            }
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *kh = k + h * head_dim;
            for (int j = 0; j < half_head; j++) {
                float k0 = kh[j];
                float k1 = kh[j + half_head];
                float c = rope_cos[j];
                float s_val = rope_sin[j];
                kh[j]             = k0 * c - k1 * s_val;
                kh[j + half_head] = k1 * c + k0 * s_val;
            }
        }

        // Multi-head attention with AVX2 vectorization
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_dim;
            float *att = s->att + h * p->seq_len;

            for (int t = 0; t <= pos; t++) {
                float *k_t = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
#if defined(__AVX2__)
                __m256 acc = _mm256_setzero_ps();
                for (int i = 0; i < head_dim; i += 8) {
                    __m256 vq = _mm256_loadu_ps(&q[i]);
                    __m256 vk = _mm256_loadu_ps(&k_t[i]);
                    acc = _mm256_fmadd_ps(vq, vk, acc);
                }
                __m128 l_v = _mm256_castps256_ps128(acc);
                __m128 hi_v = _mm256_extractf128_ps(acc, 1);
                __m128 sm_v = _mm_add_ps(l_v, hi_v);
                sm_v = _mm_hadd_ps(sm_v, sm_v);
                sm_v = _mm_hadd_ps(sm_v, sm_v);
                att[t] = _mm_cvtss_f32(sm_v) * inv_sqrt_head_dim;
#else
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k_t[i];
                }
                att[t] = score * inv_sqrt_head_dim;
#endif
            }

            softmax(att, pos + 1);

            float *xb = s->xb + h * head_dim;
            memset(xb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v_t = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float a = att[t];
#if defined(__AVX2__)
                __m256 va = _mm256_set1_ps(a);
                for (int i = 0; i < head_dim; i += 8) {
                    __m256 vxb = _mm256_loadu_ps(&xb[i]);
                    __m256 vv = _mm256_loadu_ps(&v_t[i]);
                    _mm256_storeu_ps(&xb[i], _mm256_fmadd_ps(va, vv, vxb));
                }
#else
                for (int i = 0; i < head_dim; i++) {
                    xb[i] += a * v_t[i];
                }
#endif
            }
        }

        // Output projection
        if (p->qtype == TINYLLM_QTYPE_INT8) {
            int8_t *w_wo = (int8_t *)w->wo;
            tinyllm_matmul_int8(s->xb2, w_wo + l * dim * dim, s->xb, dim, dim,
                                w->attn_scale[l * 4 + 3]);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_wo = (uint8_t *)w->wo;
            int dim_packed = (dim + 1) / 2;
            tinyllm_matmul_int4(s->xb2, w_wo + l * dim * dim_packed, s->xb, dim, dim,
                                w->attn_scale + l * (dim * 4 + kv_dim * 2) + dim * 2 + kv_dim);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            uint8_t *w_wo = (uint8_t *)w->wo;
            int dim_packed = (dim + 3) / 4;
            tinyllm_matmul_bitnet_158(s->xb2, w_wo + l * dim * dim_packed, s->xb, dim, dim,
                                     w->attn_scale + l * (dim * 4 + kv_dim * 2) + dim * 2 + kv_dim);
        } else {
            float *w_wo = (float *)w->wo;
            matmul(s->xb2, s->xb, w_wo + l * dim * dim, dim, dim);
        }

        // Residual with AVX2 vectorization
#if defined(__AVX2__)
        for (int i = 0; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&s->x[i]);
            __m256 vxb2 = _mm256_loadu_ps(&s->xb2[i]);
            _mm256_storeu_ps(&s->x[i], _mm256_add_ps(vx, vxb2));
        }
#else
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
#endif

        // FFN RMSNorm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim, 1e-5f);

        // FFN: SwiGLU
        if (p->qtype == TINYLLM_QTYPE_INT8) {
            int8_t *w_w1 = (int8_t *)w->w1;
            int8_t *w_w3 = (int8_t *)w->w3;
            tinyllm_matmul_int8(s->hb, w_w1 + l * dim * hidden_dim, s->xb, dim, hidden_dim,
                                w->ffn_scale[l * 3 + 0]);
            tinyllm_matmul_int8(s->hb2, w_w3 + l * dim * hidden_dim, s->xb, dim, hidden_dim,
                                w->ffn_scale[l * 3 + 2]);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_w1 = (uint8_t *)w->w1;
            uint8_t *w_w3 = (uint8_t *)w->w3;
            int dim_packed = (dim + 1) / 2;
            tinyllm_matmul_int4(s->hb, w_w1 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                w->ffn_scale + l * (hidden_dim * 2 + dim));
            tinyllm_matmul_int4(s->hb2, w_w3 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            uint8_t *w_w1 = (uint8_t *)w->w1;
            uint8_t *w_w3 = (uint8_t *)w->w3;
            int dim_packed = (dim + 3) / 4;
            tinyllm_matmul_bitnet_158(s->hb, w_w1 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                     w->ffn_scale + l * (hidden_dim * 2 + dim));
            tinyllm_matmul_bitnet_158(s->hb2, w_w3 + l * hidden_dim * dim_packed, s->xb, dim, hidden_dim,
                                     w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim);
        } else {
            float *w_w1 = (float *)w->w1;
            float *w_w3 = (float *)w->w3;
            matmul(s->hb, s->xb, w_w1 + l * dim * hidden_dim, dim, hidden_dim);
            matmul(s->hb2, s->xb, w_w3 + l * dim * hidden_dim, dim, hidden_dim);
        }

#if defined(__AVX2__)
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
        __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
        __m256 poly_p = _mm256_set1_ps(0.16666666666666666f);
        __m256 poly_half = _mm256_set1_ps(0.5f);
        __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
        __m256 max_x = _mm256_set1_ps(88.3762626647949f);
        __m256i bias127 = _mm256_set1_epi32(127);

        for (int i = 0; i <= hidden_dim - 8; i += 8) {
            __m256 val = _mm256_loadu_ps(&s->hb[i]);
            __m256 val2 = _mm256_loadu_ps(&s->hb2[i]);
            __m256 neg_val = _mm256_sub_ps(_mm256_setzero_ps(), val);

            __m256 z = _mm256_max_ps(neg_val, min_x);
            z = _mm256_min_ps(z, max_x);
            __m256 fx = _mm256_fmadd_ps(z, log2ef, poly_half);
            __m256i emm0 = _mm256_cvttps_epi32(_mm256_floor_ps(fx));
            __m256 n = _mm256_cvtepi32_ps(emm0);
            __m256 x_red = _mm256_fnmadd_ps(n, c1, z);
            __m256 p = _mm256_fmadd_ps(poly_p, x_red, poly_half);
            p = _mm256_fmadd_ps(p, x_red, one);
            p = _mm256_fmadd_ps(p, x_red, one);
            __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(emm0, bias127), 23);
            __m256 exp_neg = _mm256_mul_ps(p, _mm256_castsi256_ps(pow2n));

            __m256 denom = _mm256_add_ps(one, exp_neg);
            __m256 r0 = _mm256_rcp_ps(denom);
            __m256 sig = _mm256_mul_ps(r0, _mm256_fnmadd_ps(denom, r0, two));

            __m256 res = _mm256_mul_ps(_mm256_mul_ps(val, sig), val2);
            _mm256_storeu_ps(&s->hb[i], res);
        }
#else
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
#endif

        if (p->qtype == TINYLLM_QTYPE_INT8) {
            int8_t *w_w2 = (int8_t *)w->w2;
            tinyllm_matmul_int8(s->xb, w_w2 + l * hidden_dim * dim, s->hb, hidden_dim, dim,
                                w->ffn_scale[l * 3 + 1]);
        } else if (p->qtype == TINYLLM_QTYPE_INT4) {
            uint8_t *w_w2 = (uint8_t *)w->w2;
            int hidden_dim_packed = (hidden_dim + 1) / 2;
            tinyllm_matmul_int4(s->xb, w_w2 + l * dim * hidden_dim_packed, s->hb, hidden_dim, dim,
                                w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
        } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            uint8_t *w_w2 = (uint8_t *)w->w2;
            int hidden_dim_packed = (hidden_dim + 3) / 4;
            tinyllm_matmul_bitnet_158(s->xb, w_w2 + l * dim * hidden_dim_packed, s->hb, hidden_dim, dim,
                                     w->ffn_scale + l * (hidden_dim * 2 + dim) + hidden_dim * 2);
        } else {
            float *w_w2 = (float *)w->w2;
            matmul(s->xb, s->hb, w_w2 + l * hidden_dim * dim, hidden_dim, dim);
        }

        // Residual with AVX2 vectorization
#if defined(__AVX2__)
        for (int i = 0; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&s->x[i]);
            __m256 vxb = _mm256_loadu_ps(&s->xb[i]);
            _mm256_storeu_ps(&s->x[i], _mm256_add_ps(vx, vxb));
        }
#else
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
#endif
    }

    // Final RMSNorm
    rmsnorm(s->x, s->x, w->rms_final_weight, dim, 1e-5f);

    // Classifier (weight tying with token embeddings)
    if (p->qtype == TINYLLM_QTYPE_INT8) {
        int8_t *w_wcls = (int8_t *)w->wcls;
        tinyllm_matmul_int8(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale[0]);
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        uint8_t *w_wcls = (uint8_t *)w->wcls;
        tinyllm_matmul_int4(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        const int8_t *w_wcls = (const int8_t *)w->wcls;
        tinyllm_matmul_int8_rowscale(s->logits, w_wcls, s->x, dim, p->vocab_size, w->wcls_scale);
    } else {
        float *w_wcls = (float *)w->wcls;
        float *w_token_embedding = (float *)w->token_embedding;
        float *wcls = w_wcls ? w_wcls : w_token_embedding;
        matmul(s->logits, s->x, wcls, dim, p->vocab_size);
    }

    return s->logits;
}

float *tinyllm_forward(tinyllm_t *llm, tinyllm_token_t token, int pos)
{
    tinyllm_config_t *p = &llm->config;
    tinyllm_weights_t *w = &llm->weights;
    tinyllm_state_t *s = &llm->state;
    int dim = p->dim;

    // Token embedding
    if (p->qtype == TINYLLM_QTYPE_INT8) {
        int8_t *w_token_embedding = (int8_t *)w->token_embedding;
        for (int i = 0; i < dim; i++) {
            s->x[i] = w_token_embedding[token * dim + i] * w->token_embedding_scale[0];
        }
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        uint8_t *w_token_embedding = (uint8_t *)w->token_embedding;
        int dim_packed = (dim + 1) / 2;
        for (int i = 0; i < dim; i++) {
            int packed_idx = i / 2;
            int nibble = i % 2;
            uint8_t packed = w_token_embedding[token * dim_packed + packed_idx];
            int8_t val = nibble == 0 ? ((packed >> 4) & 0x0F) - 8 : (packed & 0x0F) - 8;
            s->x[i] = val * w->token_embedding_scale[token];
        }
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        const int8_t *w_token_embedding = (const int8_t *)w->token_embedding;
        const int8_t *row = w_token_embedding + (size_t)token * dim;
        float scale = w->token_embedding_scale ? w->token_embedding_scale[token] : 1.0f;
#if defined(__AVX2__)
        __m256 vscale = _mm256_set1_ps(scale);
        for (int i = 0; i <= dim - 8; i += 8) {
            __m128i r8 = _mm_loadl_epi64((const __m128i*)&row[i]);
            __m256 f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r8));
            _mm256_storeu_ps(&s->x[i], _mm256_mul_ps(f8, vscale));
        }
#else
        for (int i = 0; i < dim; i++) {
            s->x[i] = (float)row[i] * scale;
        }
#endif
    } else {
        float *w_token_embedding = (float *)w->token_embedding;
        float *content_row = w_token_embedding + token * dim;
        memcpy(s->x, content_row, dim * sizeof(float));
    }

    return tinyllm_forward_embedding(llm, s->x, pos);
}


// =============================================================================
// Sampling
// =============================================================================

tinyllm_token_t tinyllm_sample_greedy(const float *logits, int vocab_size)
{
    int max_i = 0;
    float max_v = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_v) {
            max_v = logits[i];
            max_i = i;
        }
    }
    return max_i;
}

tinyllm_token_t tinyllm_sample_temp(const float *logits, int vocab_size, float temperature)
{
    if (temperature == 0.0f) {
        return tinyllm_sample_greedy(logits, vocab_size);
    }

    // Apply temperature
    float *scaled = (float *)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        scaled[i] = logits[i] / temperature;
    }

    // Softmax
    softmax(scaled, vocab_size);

    // Sample from distribution
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += scaled[i];
        if (r < cumsum) {
            free(scaled);
            return i;
        }
    }

    free(scaled);
    return vocab_size - 1;
}

tinyllm_token_t tinyllm_sample_top_k(const float *logits, int vocab_size, int k, float temperature)
{
    if (k > 64) k = 64;
    int indices[64];
    float values[64];

    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        values[i] = -INFINITY;
    }

    for (int i = 0; i < vocab_size; i++) {
        float v = logits[i];
        if (v <= values[k - 1]) continue;

        for (int j = 0; j < k; j++) {
            if (v > values[j]) {
                // Shift down
                for (int l = k - 1; l > j; l--) {
                    values[l] = values[l - 1];
                    indices[l] = indices[l - 1];
                }
                values[j] = v;
                indices[j] = i;
                break;
            }
        }
    }

    // Apply temperature and softmax
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        values[i] = expf(values[i] / temperature);
        sum += values[i];
    }

    // Sample
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < k; i++) {
        cumsum += values[i];
        if (r < cumsum) {
            return indices[i];
        }
    }

    return indices[k - 1];
}

// =============================================================================
// Quantized MatMul
// =============================================================================

void tinyllm_matmul_int8(float *xout, const int8_t *w, const float *x, int n, int d, float scale)
{
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += w[i * n + j] * x[j];
        }
        xout[i] = sum * scale;
    }
}

void tinyllm_matmul_int4(float *xout, const uint8_t *w, const float *x, int n, int d,
                         const float *scales)
{
    int n_packed = (n + 1) / 2;
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_packed; j++) {
            uint8_t packed = w[i * n_packed + j];
            int8_t v0 = ((packed >> 4) & 0x0F) - 8;
            int8_t v1 = (packed & 0x0F) - 8;
            sum += v0 * x[j * 2];
            if (j * 2 + 1 < n) {
                sum += v1 * x[j * 2 + 1];
            }
        }
        xout[i] = sum * scales[i];
    }
}

void tinyllm_matmul_bitnet_158(float *out, const uint8_t *w, const float *in, int in_dim, int out_dim,
                               const float *scales)
{
    eif_matmul_bitnet_f32(w, scales, in, NULL, out, out_dim, in_dim);
}

#if defined(__AVX2__)
__attribute__((target("avx2,fma")))
void tinyllm_matmul_int8_rowscale(float *y, const int8_t *w, const float *x, int cols, int rows, const float *scales)
{
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r = 0; r < (rows & ~3); r += 4) {
        const int8_t *w0 = w + r * cols;
        const int8_t *w1 = w + (r + 1) * cols;
        const int8_t *w2 = w + (r + 2) * cols;
        const int8_t *w3 = w + (r + 3) * cols;

        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();

        for (int c = 0; c <= cols - 8; c += 8) {
            __m256 in = _mm256_loadu_ps(&x[c]);

            __m128i r0 = _mm_loadl_epi64((const __m128i*)&w0[c]);
            __m128i r1 = _mm_loadl_epi64((const __m128i*)&w1[c]);
            __m128i r2 = _mm_loadl_epi64((const __m128i*)&w2[c]);
            __m128i r3 = _mm_loadl_epi64((const __m128i*)&w3[c]);

            __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r0));
            __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r1));
            __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r2));
            __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r3));

            a0 = _mm256_fmadd_ps(in, f0, a0);
            a1 = _mm256_fmadd_ps(in, f1, a1);
            a2 = _mm256_fmadd_ps(in, f2, a2);
            a3 = _mm256_fmadd_ps(in, f3, a3);
        }

        #define I8_HSUM(acc, s) do { \
            __m128 l = _mm256_castps256_ps128(acc); \
            __m128 h = _mm256_extractf128_ps(acc, 1); \
            __m128 sm = _mm_add_ps(l, h); \
            sm = _mm_hadd_ps(sm, sm); \
            sm = _mm_hadd_ps(sm, sm); \
            s = _mm_cvtss_f32(sm); \
        } while(0)

        float s0, s1, s2, s3;
        I8_HSUM(a0, s0);
        I8_HSUM(a1, s1);
        I8_HSUM(a2, s2);
        I8_HSUM(a3, s3);
        y[r]     = s0 * (scales ? scales[r] : 1.0f);
        y[r + 1] = s1 * (scales ? scales[r + 1] : 1.0f);
        y[r + 2] = s2 * (scales ? scales[r + 2] : 1.0f);
        y[r + 3] = s3 * (scales ? scales[r + 3] : 1.0f);
    }

    for (int r = (rows & ~3); r < rows; r++) {
        float sum = 0.0f;
        const int8_t *wr = w + r * cols;
        for (int c = 0; c < cols; c++) {
            sum += (float)wr[c] * x[c];
        }
        y[r] = sum * (scales ? scales[r] : 1.0f);
    }
}
#elif (defined(__ARM_NEON) || defined(__aarch64__))
#include <arm_neon.h>
void tinyllm_matmul_int8_rowscale(float *y, const int8_t *w, const float *x, int cols, int rows, const float *scales)
{
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r = 0; r < (rows & ~3); r += 4) {
        const int8_t *w0 = w + r * cols;
        const int8_t *w1 = w + (r + 1) * cols;
        const int8_t *w2 = w + (r + 2) * cols;
        const int8_t *w3 = w + (r + 3) * cols;

        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);

        for (int c = 0; c <= cols - 4; c += 4) {
            float32x4_t in = vld1q_f32(&x[c]);

            int8x8_t r0 = vld1_s8(&w0[c]);
            int8x8_t r1 = vld1_s8(&w1[c]);
            int8x8_t r2 = vld1_s8(&w2[c]);
            int8x8_t r3 = vld1_s8(&w3[c]);

            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(r0))));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(r1))));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(r2))));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(r3))));

            #if defined(__ARM_FEATURE_FMA) || defined(__aarch64__)
            a0 = vfmaq_f32(a0, in, f0);
            a1 = vfmaq_f32(a1, in, f1);
            a2 = vfmaq_f32(a2, in, f2);
            a3 = vfmaq_f32(a3, in, f3);
            #else
            a0 = vaddq_f32(a0, vmulq_f32(in, f0));
            a1 = vaddq_f32(a1, vmulq_f32(in, f1));
            a2 = vaddq_f32(a2, vmulq_f32(in, f2));
            a3 = vaddq_f32(a3, vmulq_f32(in, f3));
            #endif
        }

        #if defined(__aarch64__)
        y[r]     = vaddvq_f32(a0) * (scales ? scales[r] : 1.0f);
        y[r + 1] = vaddvq_f32(a1) * (scales ? scales[r + 1] : 1.0f);
        y[r + 2] = vaddvq_f32(a2) * (scales ? scales[r + 2] : 1.0f);
        y[r + 3] = vaddvq_f32(a3) * (scales ? scales[r + 3] : 1.0f);
        #else
        y[r]     = (vgetq_lane_f32(a0, 0) + vgetq_lane_f32(a0, 1) + vgetq_lane_f32(a0, 2) + vgetq_lane_f32(a0, 3)) * (scales ? scales[r] : 1.0f);
        y[r + 1] = (vgetq_lane_f32(a1, 0) + vgetq_lane_f32(a1, 1) + vgetq_lane_f32(a1, 2) + vgetq_lane_f32(a1, 3)) * (scales ? scales[r + 1] : 1.0f);
        y[r + 2] = (vgetq_lane_f32(a2, 0) + vgetq_lane_f32(a2, 1) + vgetq_lane_f32(a2, 2) + vgetq_lane_f32(a2, 3)) * (scales ? scales[r + 2] : 1.0f);
        y[r + 3] = (vgetq_lane_f32(a3, 0) + vgetq_lane_f32(a3, 1) + vgetq_lane_f32(a3, 2) + vgetq_lane_f32(a3, 3)) * (scales ? scales[r + 3] : 1.0f);
        #endif
    }

    for (int r = (rows & ~3); r < rows; r++) {
        float sum = 0.0f;
        const int8_t *wr = w + r * cols;
        for (int c = 0; c < cols; c++) {
            sum += (float)wr[c] * x[c];
        }
        y[r] = sum * (scales ? scales[r] : 1.0f);
    }
}
#else
void tinyllm_matmul_int8_rowscale(float *y, const int8_t *w, const float *x, int cols, int rows, const float *scales)
{
    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 16384)
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const int8_t *wr = w + r * cols;
        for (int c = 0; c < cols; c++) {
            sum += (float)wr[c] * x[c];
        }
        y[r] = sum * (scales ? scales[r] : 1.0f);
    }
}
#endif

// =============================================================================
// Weight Quantization
// =============================================================================

float tinyllm_quantize_int8(const float *src, int8_t *dst, int n)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(src[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-8f)
        scale = 1e-8f;

    for (int i = 0; i < n; i++) {
        float q = src[i] / scale;
        if (q > 127.0f)
            q = 127.0f;
        if (q < -127.0f)
            q = -127.0f;
        dst[i] = (int8_t)roundf(q);
    }

    return scale;
}

/* Per-row INT8 quantization: scales[r] holds the row scale (used by the LM head). */
static void tinyllm_quantize_int8_rowscale(const float *src, int8_t *dst, int cols, int rows, float *scales)
{
    for (int r = 0; r < rows; r++) {
        scales[r] = tinyllm_quantize_int8(src + (size_t)r * cols, dst + (size_t)r * cols, cols);
    }
}

float tinyllm_quantize_int4(const float *src, uint8_t *dst, int n, int out_dim)
{
    (void)out_dim;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(src[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }

    float scale = max_abs / 7.0f;
    if (scale < 1e-8f)
        scale = 1e-8f;

    for (int i = 0; i < n; i += 2) {
        float q0 = src[i] / scale;
        float q1 = (i + 1 < n) ? src[i + 1] / scale : 0.0f;

        int8_t v0 = (int8_t)roundf(fmaxf(-8.0f, fminf(7.0f, q0))) + 8;
        int8_t v1 = (int8_t)roundf(fmaxf(-8.0f, fminf(7.0f, q1))) + 8;

        dst[i / 2] = (v0 << 4) | (v1 & 0x0F);
    }

    return scale;
}

float tinyllm_quantize_bitnet_158(const float *src, uint8_t *dst, int n, int out_dim)
{
    (void)out_dim;
    float sum_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_abs += fabsf(src[i]);
    }
    float gamma = (n > 0) ? (sum_abs / (float)n) : 1e-4f;
    if (gamma < 1e-8f)
        gamma = 1e-8f;

    int n_packed = (n + 3) / 4;
    for (int b = 0; b < n_packed; b++) {
        uint8_t byte_val = 0;
        for (int pos = 0; pos < 4; pos++) {
            int idx = b * 4 + pos;
            uint8_t code = 0;
            if (idx < n) {
                float q = roundf(src[idx] / gamma);
                if (q > 1.0f) q = 1.0f;
                if (q < -1.0f) q = -1.0f;
                int8_t val = (int8_t)q;
                if (val > 0) code = 1u;
                else if (val < 0) code = 2u;
            }
            byte_val |= (uint8_t)(code << (pos * 2));
        }
        dst[b] = byte_val;
    }

    return gamma;
}

int tinyllm_quantize_model(tinyllm_t *llm, tinyllm_qtype_t qtype, void *quant_buffer,
                           size_t buffer_size)
{
    tinyllm_config_t *p = &llm->config;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;

    // Calculate required buffer size
    size_t required = 0;
    if (qtype == TINYLLM_QTYPE_INT8) {
        required = (p->vocab_size * dim + p->n_layers * dim * dim * 4 +
                    p->n_layers * dim * kv_dim * 2 + p->n_layers * dim * hidden_dim * 3) *
                   sizeof(int8_t);
    } else if (qtype == TINYLLM_QTYPE_INT4) {
        int vocab_packed = (p->vocab_size * dim + 1) / 2;
        int dim_packed = (dim * dim + 1) / 2;
        int hidden_packed = (dim * hidden_dim + 1) / 2;
        required = (vocab_packed + p->n_layers * dim_packed * 4 +
                    p->n_layers * ((dim * kv_dim + 1) / 2) * 2 + p->n_layers * hidden_packed * 3) *
                   sizeof(uint8_t);
    } else if (qtype == TINYLLM_QTYPE_BITNET_158) {
        int dim_packed = (dim + 3) / 4;
        int hidden_packed = (hidden_dim + 3) / 4;
        required = ((size_t)p->vocab_size * dim +
                    (size_t)p->n_layers * (dim * dim_packed * 2 + kv_dim * dim_packed * 2 +
                                           hidden_dim * dim_packed * 2 + dim * hidden_packed) +
                    (size_t)p->vocab_size * dim) *
                   sizeof(uint8_t);
    }

    if (buffer_size < required) {
        fprintf(stderr, "Quant buffer too small: need %zu, got %zu\n", required, buffer_size);
        return -1;
    }

    float *w_emb = (float *)llm->weights.token_embedding;
    float *w_wq = (float *)llm->weights.wq;
    float *w_wk = (float *)llm->weights.wk;
    float *w_wv = (float *)llm->weights.wv;
    float *w_wo = (float *)llm->weights.wo;
    float *w_w1 = (float *)llm->weights.w1;
    float *w_w2 = (float *)llm->weights.w2;
    float *w_w3 = (float *)llm->weights.w3;
    /* lm_head source: separate weight if present, otherwise tied to embeddings. */
    float *w_wcls = llm->weights.wcls ? (float *)llm->weights.wcls
                                      : (float *)llm->weights.token_embedding;

    uint8_t *ptr = (uint8_t *)quant_buffer;

    if (qtype == TINYLLM_QTYPE_INT8) {
        // Allocate scale arrays
        llm->weights.token_embedding_scale = (float *)malloc(sizeof(float));
        llm->weights.attn_scale = (float *)malloc(p->n_layers * 4 * sizeof(float));
        llm->weights.ffn_scale = (float *)malloc(p->n_layers * 3 * sizeof(float));

        // Quantize embeddings
        llm->weights.token_embedding_scale[0] =
            tinyllm_quantize_int8(w_emb, (int8_t *)ptr, p->vocab_size * dim);
        llm->weights.token_embedding = ptr;
        ptr += p->vocab_size * dim * sizeof(int8_t);

        // Quantize attention weights - save base pointers
        llm->weights.wq = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 0] =
                tinyllm_quantize_int8(w_wq + l * dim * dim, (int8_t *)ptr, dim * dim);
            ptr += dim * dim * sizeof(int8_t);
        }

        llm->weights.wk = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 1] =
                tinyllm_quantize_int8(w_wk + l * dim * kv_dim, (int8_t *)ptr, dim * kv_dim);
            ptr += dim * kv_dim * sizeof(int8_t);
        }

        llm->weights.wv = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 2] =
                tinyllm_quantize_int8(w_wv + l * dim * kv_dim, (int8_t *)ptr, dim * kv_dim);
            ptr += dim * kv_dim * sizeof(int8_t);
        }

        llm->weights.wo = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.attn_scale[l * 4 + 3] =
                tinyllm_quantize_int8(w_wo + l * dim * dim, (int8_t *)ptr, dim * dim);
            ptr += dim * dim * sizeof(int8_t);
        }

        // Quantize FFN weights
        llm->weights.w1 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.ffn_scale[l * 3 + 0] =
                tinyllm_quantize_int8(w_w1 + l * dim * hidden_dim, (int8_t *)ptr, dim * hidden_dim);
            ptr += dim * hidden_dim * sizeof(int8_t);
        }

        llm->weights.w2 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.ffn_scale[l * 3 + 1] =
                tinyllm_quantize_int8(w_w2 + l * hidden_dim * dim, (int8_t *)ptr, hidden_dim * dim);
            ptr += hidden_dim * dim * sizeof(int8_t);
        }

        llm->weights.w3 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            llm->weights.ffn_scale[l * 3 + 2] =
                tinyllm_quantize_int8(w_w3 + l * dim * hidden_dim, (int8_t *)ptr, dim * hidden_dim);
            ptr += dim * hidden_dim * sizeof(int8_t);
        }

        llm->weights.wcls = llm->weights.token_embedding;
        llm->weights.wcls_scale = llm->weights.token_embedding_scale;
    } else if (qtype == TINYLLM_QTYPE_INT4) {
        // INT4: Per-row scales for each weight matrix
        int total_scales = p->vocab_size + p->n_layers * (dim * 4 + kv_dim * 2 + hidden_dim * 3);
        llm->weights.token_embedding_scale = (float *)malloc(total_scales * sizeof(float));
        llm->weights.attn_scale = llm->weights.token_embedding_scale + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * (dim * 4 + kv_dim * 2);

        // Quantize embeddings (per-row scales)
        llm->weights.token_embedding = ptr;
        int vocab_packed = (p->vocab_size * dim + 1) / 2;
        for (int i = 0; i < p->vocab_size; i++) {
            llm->weights.token_embedding_scale[i] = tinyllm_quantize_int4(
                w_emb + i * dim, (uint8_t *)ptr + i * ((dim + 1) / 2), dim, dim);
        }
        ptr += vocab_packed * sizeof(uint8_t);

        // Quantize attention weights
        llm->weights.wq = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + row] = tinyllm_quantize_int4(
                    w_wq + l * dim * dim + row * dim,
                    (uint8_t *)ptr + l * dim * ((dim + 1) / 2) + row * ((dim + 1) / 2), dim, dim);
            }
        }
        ptr += p->n_layers * dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wk = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim + row] =
                    tinyllm_quantize_int4(w_wk + l * dim * kv_dim + row * dim,
                                          (uint8_t *)ptr + l * kv_dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wv = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + row] =
                    tinyllm_quantize_int4(w_wv + l * dim * kv_dim + row * dim,
                                          (uint8_t *)ptr + l * kv_dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wo = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + kv_dim + row] =
                    tinyllm_quantize_int4(w_wo + l * dim * dim + row * dim,
                                          (uint8_t *)ptr + l * dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * dim * ((dim + 1) / 2) * sizeof(uint8_t);

        // Quantize FFN weights
        llm->weights.w1 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + row] = tinyllm_quantize_int4(
                    w_w1 + l * dim * hidden_dim + row * dim,
                    (uint8_t *)ptr + l * hidden_dim * ((dim + 1) / 2) + row * ((dim + 1) / 2), dim,
                    dim);
            }
        }
        ptr += p->n_layers * hidden_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.w2 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim + row] =
                    tinyllm_quantize_int4(w_w2 + l * hidden_dim * dim + row * hidden_dim,
                                          (uint8_t *)ptr + l * dim * ((hidden_dim + 1) / 2) +
                                              row * ((hidden_dim + 1) / 2),
                                          hidden_dim, hidden_dim);
            }
        }
        ptr += p->n_layers * dim * ((hidden_dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.w3 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim * 2 + row] =
                    tinyllm_quantize_int4(w_w3 + l * dim * hidden_dim + row * dim,
                                          (uint8_t *)ptr + l * hidden_dim * ((dim + 1) / 2) +
                                              row * ((dim + 1) / 2),
                                          dim, dim);
            }
        }
        ptr += p->n_layers * hidden_dim * ((dim + 1) / 2) * sizeof(uint8_t);

        llm->weights.wcls = llm->weights.token_embedding;
        llm->weights.wcls_scale = llm->weights.token_embedding_scale;
    } else if (qtype == TINYLLM_QTYPE_BITNET_158) {
        int total_scales = p->vocab_size * 2 + p->n_layers * (dim * 4 + kv_dim * 2 + hidden_dim * 2 + dim);
        llm->weights.token_embedding_scale = (float *)malloc(total_scales * sizeof(float));
        llm->weights.attn_scale = llm->weights.token_embedding_scale + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * (dim * 4 + kv_dim * 2);
        llm->weights.wcls_scale = llm->weights.ffn_scale + p->n_layers * (hidden_dim * 2 + dim);

        int dim_packed = (dim + 3) / 4;
        int hidden_packed = (hidden_dim + 3) / 4;

        llm->weights.token_embedding = ptr;
        tinyllm_quantize_int8_rowscale(w_emb, (int8_t *)ptr, dim, p->vocab_size,
                                       llm->weights.token_embedding_scale);
        ptr += (size_t)p->vocab_size * dim;

        llm->weights.wq = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + row] =
                    tinyllm_quantize_bitnet_158(w_wq + l * dim * dim + row * dim,
                                                ptr + l * dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.wk = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim + row] =
                    tinyllm_quantize_bitnet_158(w_wk + l * dim * kv_dim + row * dim,
                                                ptr + l * kv_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wv = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < kv_dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + row] =
                    tinyllm_quantize_bitnet_158(w_wv + l * dim * kv_dim + row * dim,
                                                ptr + l * kv_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wo = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.attn_scale[l * (dim * 4 + kv_dim * 2) + dim * 2 + kv_dim + row] =
                    tinyllm_quantize_bitnet_158(w_wo + l * dim * dim + row * dim,
                                                ptr + l * dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.w1 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + row] =
                    tinyllm_quantize_bitnet_158(w_w1 + l * dim * hidden_dim + row * dim,
                                                ptr + l * hidden_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * hidden_dim * dim_packed;

        llm->weights.w2 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim * 2 + row] =
                    tinyllm_quantize_bitnet_158(w_w2 + l * hidden_dim * dim + row * hidden_dim,
                                                ptr + l * dim * hidden_packed + row * hidden_packed,
                                                hidden_dim, hidden_dim);
            }
        }
        ptr += p->n_layers * dim * hidden_packed;

        llm->weights.w3 = ptr;
        for (int l = 0; l < p->n_layers; l++) {
            for (int row = 0; row < hidden_dim; row++) {
                llm->weights.ffn_scale[l * (hidden_dim * 2 + dim) + hidden_dim + row] =
                    tinyllm_quantize_bitnet_158(w_w3 + l * dim * hidden_dim + row * dim,
                                                ptr + l * hidden_dim * dim_packed + row * dim_packed, dim, dim);
            }
        }
        ptr += p->n_layers * hidden_dim * dim_packed;

        /* Output classifier (lm_head): INT8 with per-row scale */
        llm->weights.wcls = ptr;
        tinyllm_quantize_int8_rowscale(w_wcls, (int8_t *)ptr, dim, p->vocab_size,
                                       llm->weights.wcls_scale);
        ptr += (size_t)p->vocab_size * dim;
    }

    llm->config.qtype = qtype;
    printf("Model quantized to %s\n",
           qtype == TINYLLM_QTYPE_BITNET_158 ? "BitNet b1.58 (Ternary 2-bit)" :
           qtype == TINYLLM_QTYPE_INT8 ? "INT8" : "INT4");
    printf("Quantized size: %zu bytes (%.1f MB)\n", required, required / 1024.0f / 1024.0f);
    printf("Compression ratio: %.1fx\n",
           (float)(p->vocab_size * dim +
                   p->n_layers * (dim * dim * 4 + dim * kv_dim * 2 + dim * hidden_dim * 3)) *
               sizeof(float) / required);

    return 0;
}

// =============================================================================
// Model Save/Load
// =============================================================================

#define TINYLLM_FILE_MAGIC 0x54544D4C // "TTML" in hex
/* v2: BitNet-158 uses INT8 (per-row scaled) tied embeddings/LM head + packed layers.
 * Bump invalidates v1 artifacts written with the old packed-embedding layout. */
#define TINYLLM_FILE_VERSION 2

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t config_size;
    uint32_t weights_size;
    uint32_t scales_size;
    uint32_t rms_weights_size;
    uint32_t reserved[3];
} tinyllm_file_header_t;

// Forward declaration
static void tinyllm_setup_quantized_pointers(tinyllm_t *llm, uint8_t *weights_ptr,
                                             float *scales_ptr, size_t weights_size);

int tinyllm_save_quantized_model(const tinyllm_t *llm, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to create file: %s\n", filename);
        return -1;
    }

    const tinyllm_config_t *p = &llm->config;

    // Calculate sizes
    size_t config_size = sizeof(tinyllm_config_t);
    size_t weights_size = 0;
    size_t scales_size = 0;
    size_t rms_weights_size =
        (p->n_layers * p->dim * 2 + p->dim) * sizeof(float); // rms_att + rms_ffn + rms_final

    if (p->qtype == TINYLLM_QTYPE_INT8) {
        weights_size = (p->vocab_size * p->dim + p->n_layers * p->dim * p->dim * 4 +
                        p->n_layers * p->dim * ((p->dim * p->n_kv_heads) / p->n_heads) * 2 +
                        p->n_layers * p->dim * p->hidden_dim * 3) *
                       sizeof(int8_t);
        scales_size = (1 + p->n_layers * 7) * sizeof(float);
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        weights_size =
            ((p->vocab_size * p->dim + 1) / 2 + p->n_layers * p->dim * ((p->dim + 1) / 2) * 4 +
             p->n_layers * kv_dim * ((p->dim + 1) / 2) * 2 +
             p->n_layers * p->hidden_dim * ((p->dim + 1) / 2) * 2 +
             p->n_layers * p->dim * ((p->hidden_dim + 1) / 2)) *
            sizeof(uint8_t);
        scales_size =
            (p->vocab_size + p->n_layers * (p->dim * 4 + kv_dim * 2 + p->hidden_dim * 2 + p->dim)) *
            sizeof(float);
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        int dim_packed = (p->dim + 3) / 4;
        int hidden_packed = (p->hidden_dim + 3) / 4;
        weights_size = ((size_t)p->vocab_size * p->dim +
                        (size_t)p->n_layers * (p->dim * dim_packed * 2 + kv_dim * dim_packed * 2 +
                                               p->hidden_dim * dim_packed * 2 + p->dim * hidden_packed) +
                        (size_t)p->vocab_size * p->dim) *
                       sizeof(uint8_t);
        scales_size =
            (p->vocab_size * 2 + p->n_layers * (p->dim * 4 + kv_dim * 2 + p->hidden_dim * 2 + p->dim)) *
            sizeof(float);
    } else {
        fprintf(stderr, "Only quantized models can be saved\n");
        fclose(f);
        return -1;
    }

    // Write header
    tinyllm_file_header_t header = {.magic = TINYLLM_FILE_MAGIC,
                                    .version = TINYLLM_FILE_VERSION,
                                    .config_size = (uint32_t)config_size,
                                    .weights_size = (uint32_t)weights_size,
                                    .scales_size = (uint32_t)scales_size,
                                    .rms_weights_size = (uint32_t)rms_weights_size,
                                    .reserved = {0}};

    if (fwrite(&header, sizeof(header), 1, f) != 1)
        goto error;

    // Write config
    if (fwrite(p, config_size, 1, f) != 1)
        goto error;

    // Write weights (pointed to by llm->weights)
    const uint8_t *weights_data = (const uint8_t *)llm->weights.token_embedding;
    if (fwrite(weights_data, weights_size, 1, f) != 1)
        goto error;

    // Write scales - each scale array separately (they may not be contiguous)
    if (p->qtype == TINYLLM_QTYPE_INT8) {
        if (fwrite(llm->weights.token_embedding_scale, sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.attn_scale, p->n_layers * 4 * sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.ffn_scale, p->n_layers * 3 * sizeof(float), 1, f) != 1)
            goto error;
    } else {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        if (fwrite(llm->weights.token_embedding_scale, p->vocab_size * sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.attn_scale, p->n_layers * (p->dim * 4 + kv_dim * 2) * sizeof(float), 1, f) != 1)
            goto error;
        if (fwrite(llm->weights.ffn_scale, p->n_layers * (p->hidden_dim * 2 + p->dim) * sizeof(float), 1, f) != 1)
            goto error;
        if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
            if (fwrite(llm->weights.wcls_scale, p->vocab_size * sizeof(float), 1, f) != 1)
                goto error;
        }
    }

    // Write RMS weights (not quantized)
    if (fwrite(llm->weights.rms_att_weight, p->n_layers * p->dim * sizeof(float), 1, f) != 1)
        goto error;
    if (fwrite(llm->weights.rms_ffn_weight, p->n_layers * p->dim * sizeof(float), 1, f) != 1)
        goto error;
    if (fwrite(llm->weights.rms_final_weight, p->dim * sizeof(float), 1, f) != 1)
        goto error;

    fclose(f);

    printf("Model saved to %s\n", filename);
    printf("  Config: %zu bytes\n", config_size);
    printf("  Weights: %zu bytes\n", weights_size);
    printf("  Scales: %zu bytes\n", scales_size);
    printf("  RMS weights: %zu bytes\n", rms_weights_size);
    printf("  Total: %zu bytes (%.1f MB)\n",
           sizeof(header) + config_size + weights_size + scales_size + rms_weights_size,
           (sizeof(header) + config_size + weights_size + scales_size + rms_weights_size) /
               1024.0f / 1024.0f);

    return 0;

error:
    fprintf(stderr, "Error writing model file\n");
    fclose(f);
    return -1;
}

int tinyllm_load_quantized_model(tinyllm_t *llm, const char *filename, void *quant_buffer,
                                 size_t buffer_size)
{
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return -1;
    }

    // Read and verify header
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Error reading file magic\n");
        fclose(f);
        return -1;
    }
    fseek(f, 0, SEEK_SET);

    size_t weights_size = 0, scales_size = 0, rms_weights_size = 0;
    long weights_file_offset = 0, scales_file_offset = 0, rms_file_offset = 0;

    if (magic == 0x53564C4D) { // 'SVLM' (Multimodal / Granite Docling format)
        uint32_t svlm_hdr[12];
        if (fread(svlm_hdr, sizeof(uint32_t), 12, f) != 12) {
            fprintf(stderr, "Error reading SVLM header\n");
            fclose(f);
            return -1;
        }
        weights_file_offset = (long)svlm_hdr[3];
        scales_file_offset  = (long)svlm_hdr[4];
        rms_file_offset     = (long)svlm_hdr[5];
        weights_size        = svlm_hdr[7];
        scales_size         = svlm_hdr[8];
        rms_weights_size    = svlm_hdr[9];
    } else if (magic == TINYLLM_FILE_MAGIC || magic == 0x544C4C4D) { // 'TTML' or 'TLLM'
        tinyllm_file_header_t header;
        if (fread(&header, sizeof(header), 1, f) != 1) {
            fprintf(stderr, "Error reading TTML header\n");
            fclose(f);
            return -1;
        }
        weights_file_offset = sizeof(header) + header.config_size;
        scales_file_offset  = weights_file_offset + header.weights_size;
        rms_file_offset     = scales_file_offset + header.scales_size;
        weights_size        = header.weights_size;
        scales_size         = header.scales_size;
        rms_weights_size    = header.rms_weights_size;
    } else {
        fprintf(stderr, "Invalid file format (wrong magic: 0x%08X)\n", magic);
        fclose(f);
        return -1;
    }

    size_t total_size = weights_size + scales_size + rms_weights_size;
    if (buffer_size < total_size) {
        fprintf(stderr, "Buffer too small: need %zu, got %zu\n", total_size, buffer_size);
        fclose(f);
        return -1;
    }

    // Read weights into buffer
    uint8_t *ptr = (uint8_t *)quant_buffer;
    fseek(f, weights_file_offset, SEEK_SET);
    if (fread(ptr, weights_size, 1, f) != 1) {
        fprintf(stderr, "Error reading weights\n");
        fclose(f);
        return -1;
    }

    // Read scales into buffer after weights
    float *scales_ptr = (float *)(ptr + weights_size);
    fseek(f, scales_file_offset, SEEK_SET);
    if (fread(scales_ptr, scales_size, 1, f) != 1) {
        fprintf(stderr, "Error reading scales\n");
        fclose(f);
        return -1;
    }

    // Read RMS weights into buffer after scales
    float *rms_ptr = (float *)((uint8_t *)scales_ptr + scales_size);
    fseek(f, rms_file_offset, SEEK_SET);
    if (fread(rms_ptr, rms_weights_size, 1, f) != 1) {
        fprintf(stderr, "Error reading RMS weights\n");
        fclose(f);
        return -1;
    }

    fclose(f);

    // Set up all pointers
    tinyllm_config_t *p = &llm->config;

    // RMS weights pointers
    llm->weights.rms_att_weight = rms_ptr;
    llm->weights.rms_ffn_weight = rms_ptr + p->n_layers * p->dim;
    llm->weights.rms_final_weight = llm->weights.rms_ffn_weight + p->n_layers * p->dim;

    // Quantized weights pointers
    tinyllm_setup_quantized_pointers(llm, ptr, scales_ptr, weights_size);

    printf("Model loaded from %s\n", filename);
    const char *qstr = "FP32";
    if (llm->config.qtype == TINYLLM_QTYPE_INT8) qstr = "INT8";
    else if (llm->config.qtype == TINYLLM_QTYPE_INT4) qstr = "INT4";
    else if (llm->config.qtype == TINYLLM_QTYPE_BITNET_158) qstr = "BitNet b1.58 (Ternary 1.58-bit)";
    printf("  Quantization: %s\n", qstr);

    return 0;
}

static void tinyllm_setup_quantized_pointers(tinyllm_t *llm, uint8_t *weights_ptr,
                                             float *scales_ptr, size_t weights_size)
{
    (void)weights_size; // Unused for now
    tinyllm_config_t *p = &llm->config;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;

    if (p->qtype == TINYLLM_QTYPE_INT8) {
        uint8_t *ptr = weights_ptr;

        llm->weights.token_embedding = ptr;
        llm->weights.token_embedding_scale = scales_ptr;
        ptr += p->vocab_size * dim * sizeof(int8_t);

        llm->weights.wq = ptr;
        ptr += p->n_layers * dim * dim * sizeof(int8_t);

        llm->weights.wk = ptr;
        ptr += p->n_layers * dim * kv_dim * sizeof(int8_t);

        llm->weights.wv = ptr;
        ptr += p->n_layers * dim * kv_dim * sizeof(int8_t);

        llm->weights.wo = ptr;
        ptr += p->n_layers * dim * dim * sizeof(int8_t);

        llm->weights.w1 = ptr;
        ptr += p->n_layers * dim * hidden_dim * sizeof(int8_t);

        llm->weights.w2 = ptr;
        ptr += p->n_layers * hidden_dim * dim * sizeof(int8_t);

        llm->weights.w3 = ptr;

        llm->weights.wcls = llm->weights.token_embedding;
        llm->weights.wcls_scale = llm->weights.token_embedding_scale;

        // Set up attention and FFN scales
        llm->weights.attn_scale = scales_ptr + 1;                  // After token_embedding_scale
        llm->weights.ffn_scale = scales_ptr + 1 + p->n_layers * 4; // After attn_scale
    } else if (p->qtype == TINYLLM_QTYPE_INT4) {
        uint8_t *ptr = weights_ptr;
        int dim_packed = (dim + 1) / 2;
        int hidden_packed = (hidden_dim + 1) / 2;

        llm->weights.token_embedding = ptr;
        llm->weights.token_embedding_scale = scales_ptr;
        ptr += p->vocab_size * dim_packed;

        llm->weights.wq = ptr;
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.wk = ptr;
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wv = ptr;
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wo = ptr;
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.w1 = ptr;
        ptr += p->n_layers * hidden_dim * dim_packed;

        llm->weights.w2 = ptr;
        ptr += p->n_layers * dim * hidden_packed;

        llm->weights.w3 = ptr;

        llm->weights.wcls = llm->weights.token_embedding;
        llm->weights.wcls_scale = llm->weights.token_embedding_scale;

        llm->weights.attn_scale = scales_ptr + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * (dim * 4 + kv_dim * 2);
    } else if (p->qtype == TINYLLM_QTYPE_BITNET_158) {
        uint8_t *ptr = weights_ptr;
        int dim_packed = (dim + 3) / 4;
        int hidden_packed = (hidden_dim + 3) / 4;

        // 1. Token embeddings: INT8 [vocab_size, dim]
        llm->weights.token_embedding = ptr;
        ptr += (size_t)p->vocab_size * dim * sizeof(int8_t);

        // 2. Transformer layers: BitNet b1.58 Ternary 2-bit
        llm->weights.wq = ptr;
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.wk = ptr;
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wv = ptr;
        ptr += p->n_layers * kv_dim * dim_packed;

        llm->weights.wo = ptr;
        ptr += p->n_layers * dim * dim_packed;

        llm->weights.w1 = ptr;
        ptr += p->n_layers * hidden_dim * dim_packed;

        llm->weights.w2 = ptr;
        ptr += p->n_layers * dim * hidden_packed;

        llm->weights.w3 = ptr;
        ptr += p->n_layers * hidden_dim * dim_packed;

        // 3. Output classifier (lm_head): INT8 [vocab_size, dim]
        llm->weights.wcls = ptr;
        ptr += (size_t)p->vocab_size * dim * sizeof(int8_t);

        // 4. Scales layout
        llm->weights.token_embedding_scale = scales_ptr;
        llm->weights.attn_scale = scales_ptr + p->vocab_size;
        llm->weights.ffn_scale = llm->weights.attn_scale + p->n_layers * (dim * 4 + kv_dim * 2);
        llm->weights.wcls_scale = llm->weights.ffn_scale + p->n_layers * (hidden_dim * 2 + dim);
    }
}

// =============================================================================
// Utility
// =============================================================================

void tinyllm_print_info(const tinyllm_t *llm)
{
    printf("TinyLLM Model:\n");
    printf("  Dim: %d\n", llm->config.dim);
    printf("  Hidden: %d\n", llm->config.hidden_dim);
    printf("  Layers: %d\n", llm->config.n_layers);
    printf("  Heads: %d\n", llm->config.n_heads);
    printf("  Vocab: %d\n", llm->config.vocab_size);
    printf("  SeqLen: %d\n", llm->config.seq_len);
    printf("  Quantization: %s\n", llm->config.qtype == TINYLLM_QTYPE_BITNET_158 ? "BitNet b1.58 (Ternary 2-bit)"
                                   : llm->config.qtype == TINYLLM_QTYPE_INT4   ? "INT4"
                                   : llm->config.qtype == TINYLLM_QTYPE_INT8   ? "INT8"
                                                                               : "FP32");
    printf("  Memory: %zu bytes\n", llm->memory_size);
}

void tinyllm_memory_stats(const tinyllm_t *llm, size_t *used, size_t *total)
{
    *total = llm->memory_size;
    *used = tinyllm_memory_size(&llm->config);
}

// =============================================================================
// Tokenizer Loader
// =============================================================================

int tinyllm_load_tokenizer(tinyllm_t *llm, const char *tokenizer_path)
{
    if (!llm || !tokenizer_path)
        return -1;

    FILE *f = fopen(tokenizer_path, "rb");
    if (!f)
        return -1;

    int max_token_length = 0;
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    int vocab_size = llm->config.vocab_size;
    if (vocab_size <= 0) {
        fclose(f);
        return -1;
    }

    llm->tokenizer.vocab = (char **)malloc(vocab_size * sizeof(char *));
    llm->tokenizer.vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    if (!llm->tokenizer.vocab || !llm->tokenizer.vocab_scores) {
        fclose(f);
        return -1;
    }
    llm->tokenizer.vocab_size = vocab_size;
    llm->tokenizer.bos_token = 1;
    llm->tokenizer.eos_token = 2;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(&llm->tokenizer.vocab_scores[i], sizeof(float), 1, f) != 1) {
            fclose(f);
            return -1;
        }
        int len = 0;
        if (fread(&len, sizeof(int), 1, f) != 1 || len < 0 || len > 2048) {
            fclose(f);
            return -1;
        }
        llm->tokenizer.vocab[i] = (char *)malloc(len + 1);
        if (!llm->tokenizer.vocab[i]) {
            fclose(f);
            return -1;
        }
        if (fread(llm->tokenizer.vocab[i], 1, len, f) != (size_t)len) {
            fclose(f);
            return -1;
        }
        llm->tokenizer.vocab[i][len] = '\0';
    }

    fclose(f);
    return 0;
}

void tinyllm_free_tokenizer(tinyllm_t *llm)
{
    if (!llm || !llm->tokenizer.vocab)
        return;
    for (int i = 0; i < llm->tokenizer.vocab_size; i++) {
        if (llm->tokenizer.vocab[i]) {
            free(llm->tokenizer.vocab[i]);
            llm->tokenizer.vocab[i] = NULL;
        }
    }
    free(llm->tokenizer.vocab);
    free(llm->tokenizer.vocab_scores);
    llm->tokenizer.vocab = NULL;
    llm->tokenizer.vocab_scores = NULL;
    llm->tokenizer.vocab_size = 0;
}

