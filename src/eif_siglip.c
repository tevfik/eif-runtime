/**
 * @file eif_siglip.c
 * @brief Native Pure C99 SigLIP2 Vision Encoder & Connector Implementation.
 */

#include "eif_siglip.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIGLIP_PATCH_SIZE   16
#define SIGLIP_IMG_SIZE     512
#define SIGLIP_NUM_LAYERS   12
#define SIGLIP_HIDDEN_SIZE  768
#define SIGLIP_NUM_HEADS    12
#define SIGLIP_HEAD_DIM     64
#define SIGLIP_INTERM_SIZE  3072
#define SIGLIP_NUM_PATCHES  1024
#define SIGLIP_CONN_IN      12288
#define SIGLIP_CONN_OUT     576
#define SIGLIP_NUM_TOKENS   64

int eif_siglip_init_from_buffer(eif_siglip_t *model, const void *buffer, size_t buffer_size)
{
    if (!model || !buffer) return -1;
    memset(model, 0, sizeof(*model));

    if (buffer_size < sizeof(eif_siglip_header_t)) {
        fprintf(stderr, "[EIF SigLIP Error] Buffer too small for header\n");
        return -2;
    }

    const eif_siglip_header_t *hdr = (const eif_siglip_header_t *)buffer;
    if (hdr->magic != EIF_SIGLIP_MAGIC) {
        fprintf(stderr, "[EIF SigLIP Error] Invalid magic: 0x%08X (expected 0x%08X)\n",
                hdr->magic, EIF_SIGLIP_MAGIC);
        return -3;
    }

    model->header = *hdr;

    const uint8_t *ptr = (const uint8_t *)buffer + sizeof(eif_siglip_header_t);

    /* 1. Embeddings */
    model->pe_w = (const float *)ptr;
    ptr += (size_t)SIGLIP_HIDDEN_SIZE * 3 * SIGLIP_PATCH_SIZE * SIGLIP_PATCH_SIZE * sizeof(float);

    model->pe_b = (const float *)ptr;
    ptr += (size_t)SIGLIP_HIDDEN_SIZE * sizeof(float);

    model->pos_w = (const float *)ptr;
    ptr += (size_t)SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float);

    /* 2. 12 Transformer Layers */
    for (int l = 0; l < SIGLIP_NUM_LAYERS; l++) {
        eif_siglip_layer_t *layer = &model->layers[l];

        layer->ln1_w = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->ln1_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->q_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->q_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->k_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->k_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->v_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->v_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->out_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->out_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->ln2_w = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->ln2_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

        layer->fc1_w = (const float *)ptr; ptr += (size_t)SIGLIP_INTERM_SIZE * SIGLIP_HIDDEN_SIZE * sizeof(float);
        layer->fc1_b = (const float *)ptr; ptr += SIGLIP_INTERM_SIZE * sizeof(float);

        layer->fc2_w = (const float *)ptr; ptr += (size_t)SIGLIP_HIDDEN_SIZE * SIGLIP_INTERM_SIZE * sizeof(float);
        layer->fc2_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
    }

    /* 3. Post LayerNorm */
    model->post_ln_w = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);
    model->post_ln_b = (const float *)ptr; ptr += SIGLIP_HIDDEN_SIZE * sizeof(float);

    /* 4. Connector */
    model->conn_w = (const float *)ptr; ptr += (size_t)SIGLIP_CONN_OUT * SIGLIP_CONN_IN * sizeof(float);

    size_t consumed = (size_t)(ptr - (const uint8_t *)buffer);
    if (consumed > buffer_size) {
        fprintf(stderr, "[EIF SigLIP Error] Expected at least %zu bytes, got %zu\n", consumed, buffer_size);
        return -4;
    }

    model->is_initialized = true;
    return 0;
}

int eif_siglip_load_from_file(eif_siglip_t *model, const char *filepath, size_t offset, size_t size)
{
    if (!model || !filepath) return -1;
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[EIF SigLIP Error] Failed to open %s\n", filepath);
        return -1;
    }

    if (fseek(f, (long)offset, SEEK_SET) != 0) {
        fclose(f);
        return -2;
    }

    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) {
        fclose(f);
        return -3;
    }

    if (fread(buf, 1, size, f) != size) {
        free(buf);
        fclose(f);
        return -4;
    }
    fclose(f);

    int rc = eif_siglip_init_from_buffer(model, buf, size);
    if (rc != 0) {
        free(buf);
        return rc;
    }

    model->allocated_weights = buf;
    return 0;
}

void eif_siglip_free(eif_siglip_t *model)
{
    if (!model) return;
    if (model->allocated_weights) {
        free(model->allocated_weights);
        model->allocated_weights = NULL;
    }
    model->is_initialized = false;
}

static void layer_norm_f32(const float *x, const float *w, const float *b, float *out, int n, float eps)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    float mean = sum / (float)n;

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        sum_sq += diff * diff;
    }
    float var = sum_sq / (float)n;
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

#ifdef _OPENMP
#include <omp.h>
#endif

int eif_siglip_encode_tile(const eif_siglip_t *model, const float *tile_pixels, float *out_tokens)
{
    if (!model || !model->is_initialized || !tile_pixels || !out_tokens) return -1;

    /* Temporary scratch buffers */
    float *x = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *x_norm = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *q = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *k = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *v = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *attn_out = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_HIDDEN_SIZE * sizeof(float));
    float *mlp_h1 = (float *)malloc(SIGLIP_NUM_PATCHES * SIGLIP_INTERM_SIZE * sizeof(float));
    float *grouped = (float *)malloc(SIGLIP_NUM_TOKENS * SIGLIP_CONN_IN * sizeof(float));

    if (!x || !x_norm || !q || !k || !v || !attn_out || !mlp_h1 || !grouped) {
        if (x) free(x);
        if (x_norm) free(x_norm);
        if (q) free(q);
        if (k) free(k);
        if (v) free(v);
        if (attn_out) free(attn_out);
        if (mlp_h1) free(mlp_h1);
        if (grouped) free(grouped);
        return -2;
    }

    /* 1. Patch Embedding Conv2D + Position Embedding */
    /* tile_pixels: [3, 512, 512], pe_w: [768, 3, 16, 16] */
    const int k_spatial = 3 * SIGLIP_PATCH_SIZE * SIGLIP_PATCH_SIZE; /* 3 * 256 = 768 */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int py = 0; py < 32; py++) {
        for (int px = 0; px < 32; px++) {
            int p_idx = py * 32 + px;
            float *out_p = x + (size_t)p_idx * SIGLIP_HIDDEN_SIZE;
            const float *pos_p = model->pos_w + (size_t)p_idx * SIGLIP_HIDDEN_SIZE;

            for (int d = 0; d < SIGLIP_HIDDEN_SIZE; d++) {
                float val = model->pe_b[d];
                const float *w_kernel = model->pe_w + (size_t)d * k_spatial;

                int w_idx = 0;
                for (int c = 0; c < 3; c++) {
                    const float *c_pixels = tile_pixels + (size_t)c * (SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE);
                    for (int ky = 0; ky < SIGLIP_PATCH_SIZE; ky++) {
                        const float *row = c_pixels + (size_t)(py * SIGLIP_PATCH_SIZE + ky) * SIGLIP_IMG_SIZE + (px * SIGLIP_PATCH_SIZE);
                        for (int kx = 0; kx < SIGLIP_PATCH_SIZE; kx++) {
                            val += row[kx] * w_kernel[w_idx++];
                        }
                    }
                }
                out_p[d] = val + pos_p[d];
            }
        }
    }

    /* 2. 12 Transformer Layers */
    for (int l = 0; l < SIGLIP_NUM_LAYERS; l++) {
        const eif_siglip_layer_t *layer = &model->layers[l];

        /* LayerNorm 1 */
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            layer_norm_f32(x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           layer->ln1_w, layer->ln1_b,
                           x_norm + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           SIGLIP_HIDDEN_SIZE, 1e-6f);
        }

        /* Q, K, V projections */
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            const float *xn = x_norm + (size_t)p * SIGLIP_HIDDEN_SIZE;
            float *qp = q + (size_t)p * SIGLIP_HIDDEN_SIZE;
            float *kp = k + (size_t)p * SIGLIP_HIDDEN_SIZE;
            float *vp = v + (size_t)p * SIGLIP_HIDDEN_SIZE;

            for (int d = 0; d < SIGLIP_HIDDEN_SIZE; d++) {
                const float *qw = layer->q_w + (size_t)d * SIGLIP_HIDDEN_SIZE;
                const float *kw = layer->k_w + (size_t)d * SIGLIP_HIDDEN_SIZE;
                const float *vw = layer->v_w + (size_t)d * SIGLIP_HIDDEN_SIZE;

                float q_val = layer->q_b[d];
                float k_val = layer->k_b[d];
                float v_val = layer->v_b[d];

                for (int m = 0; m < SIGLIP_HIDDEN_SIZE; m++) {
                    float xm = xn[m];
                    q_val += xm * qw[m];
                    k_val += xm * kw[m];
                    v_val += xm * vw[m];
                }

                qp[d] = q_val;
                kp[d] = k_val;
                vp[d] = v_val;
            }
        }

        /* Multi-Head Attention: 12 heads, head_dim 64, scale = 1/8 = 0.125f */
        const float attn_scale = 1.0f / sqrtf((float)SIGLIP_HEAD_DIM);
        #pragma omp parallel for schedule(dynamic)
        for (int h = 0; h < SIGLIP_NUM_HEADS; h++) {
            int h_off = h * SIGLIP_HEAD_DIM;
            float local_scores[SIGLIP_NUM_PATCHES];

            for (int i = 0; i < SIGLIP_NUM_PATCHES; i++) {
                const float *qi = q + (size_t)i * SIGLIP_HIDDEN_SIZE + h_off;

                float max_s = -1e30f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    const float *kj = k + (size_t)j * SIGLIP_HIDDEN_SIZE + h_off;
                    float dot = 0.0f;
                    for (int d = 0; d < SIGLIP_HEAD_DIM; d++) {
                        dot += qi[d] * kj[d];
                    }
                    float s = dot * attn_scale;
                    local_scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                /* Softmax */
                float sum_exp = 0.0f;
                for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                    float ev = expf(local_scores[j] - max_s);
                    local_scores[j] = ev;
                    sum_exp += ev;
                }
                float inv_sum = 1.0f / sum_exp;

                /* Output projection accumulator */
                float *out_ih = attn_out + (size_t)i * SIGLIP_HIDDEN_SIZE + h_off;
                for (int d = 0; d < SIGLIP_HEAD_DIM; d++) {
                    float o = 0.0f;
                    for (int j = 0; j < SIGLIP_NUM_PATCHES; j++) {
                        o += local_scores[j] * v[(size_t)j * SIGLIP_HIDDEN_SIZE + h_off + d];
                    }
                    out_ih[d] = o * inv_sum;
                }
            }
        }

        /* Out Projection + Residual Add */
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            const float *ao = attn_out + (size_t)p * SIGLIP_HIDDEN_SIZE;
            float *xp = x + (size_t)p * SIGLIP_HIDDEN_SIZE;

            for (int d = 0; d < SIGLIP_HIDDEN_SIZE; d++) {
                const float *ow = layer->out_w + (size_t)d * SIGLIP_HIDDEN_SIZE;
                float val = layer->out_b[d];
                for (int m = 0; m < SIGLIP_HIDDEN_SIZE; m++) {
                    val += ao[m] * ow[m];
                }
                xp[d] += val;
            }
        }

        /* LayerNorm 2 */
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            layer_norm_f32(x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           layer->ln2_w, layer->ln2_b,
                           x_norm + (size_t)p * SIGLIP_HIDDEN_SIZE,
                           SIGLIP_HIDDEN_SIZE, 1e-6f);
        }

        /* MLP FC1 + GELU (tanh approximate) */
        const float sqrt_2_over_pi = 0.7978845608028654f;
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            const float *xn = x_norm + (size_t)p * SIGLIP_HIDDEN_SIZE;
            float *h1 = mlp_h1 + (size_t)p * SIGLIP_INTERM_SIZE;

            for (int d = 0; d < SIGLIP_INTERM_SIZE; d++) {
                const float *fc1w = layer->fc1_w + (size_t)d * SIGLIP_HIDDEN_SIZE;
                float val = layer->fc1_b[d];
                for (int m = 0; m < SIGLIP_HIDDEN_SIZE; m++) {
                    val += xn[m] * fc1w[m];
                }
                float u = sqrt_2_over_pi * (val + 0.044715f * val * val * val);
                h1[d] = 0.5f * val * (1.0f + tanhf(u));
            }
        }

        /* MLP FC2 + Residual Add */
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
            const float *h1 = mlp_h1 + (size_t)p * SIGLIP_INTERM_SIZE;
            float *xp = x + (size_t)p * SIGLIP_HIDDEN_SIZE;

            for (int d = 0; d < SIGLIP_HIDDEN_SIZE; d++) {
                const float *fc2w = layer->fc2_w + (size_t)d * SIGLIP_INTERM_SIZE;
                float val = layer->fc2_b[d];
                for (int m = 0; m < SIGLIP_INTERM_SIZE; m++) {
                    val += h1[m] * fc2w[m];
                }
                xp[d] += val;
            }
        }
    }

    /* 3. Post LayerNorm */
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < SIGLIP_NUM_PATCHES; p++) {
        layer_norm_f32(x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                       model->post_ln_w, model->post_ln_b,
                       x + (size_t)p * SIGLIP_HIDDEN_SIZE,
                       SIGLIP_HIDDEN_SIZE, 1e-6f);
    }

    /* 4. Spatial Patch Merging: 32x32 -> 8x8 blocks of 4x4 patches (16 * 768 = 12,288) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int by = 0; by < 8; by++) {
        for (int bx = 0; bx < 8; bx++) {
            int block_idx = by * 8 + bx;
            float *grp = grouped + (size_t)block_idx * SIGLIP_CONN_IN;

            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    int p_idx = (by * 4 + py) * 32 + (bx * 4 + px);
                    int local_p = py * 4 + px;
                    memcpy(grp + (size_t)local_p * SIGLIP_HIDDEN_SIZE,
                           x + (size_t)p_idx * SIGLIP_HIDDEN_SIZE,
                           SIGLIP_HIDDEN_SIZE * sizeof(float));
                }
            }
        }
    }

    /* 5. Connector Modality Projection: [64, 12288] @ W_conn.T [12288, 576] -> [64, 576] */
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < SIGLIP_NUM_TOKENS; b++) {
        const float *grp = grouped + (size_t)b * SIGLIP_CONN_IN;
        float *tok = out_tokens + (size_t)b * SIGLIP_CONN_OUT;

        for (int d = 0; d < SIGLIP_CONN_OUT; d++) {
            const float *cw = model->conn_w + (size_t)d * SIGLIP_CONN_IN;
            float sum = 0.0f;
            for (int m = 0; m < SIGLIP_CONN_IN; m++) {
                sum += grp[m] * cw[m];
            }
            tok[d] = sum;
        }
    }

    free(x);
    free(x_norm);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(mlp_h1);
    free(grouped);

    return 0;
}

static void resize_bilinear_rgb(const uint8_t *src, int src_w, int src_h,
                                float *dst_chw, int dst_w, int dst_h)
{
    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    for (int y = 0; y < dst_h; y++) {
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(src_y);
        int y1 = y0 + 1;
        float dy = src_y - (float)y0;
        if (y0 < 0) { y0 = 0; dy = 0.0f; }
        if (y1 >= src_h) { y1 = src_h - 1; }

        for (int x = 0; x < dst_w; x++) {
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(src_x);
            int x1 = x0 + 1;
            float dx = src_x - (float)x0;
            if (x0 < 0) { x0 = 0; dx = 0.0f; }
            if (x1 >= src_w) { x1 = src_w - 1; }

            for (int c = 0; c < 3; c++) {
                float v00 = (float)src[(y0 * src_w + x0) * 3 + c];
                float v01 = (float)src[(y0 * src_w + x1) * 3 + c];
                float v10 = (float)src[(y1 * src_w + x0) * 3 + c];
                float v11 = (float)src[(y1 * src_w + x1) * 3 + c];
                float val = (1.0f - dy) * ((1.0f - dx) * v00 + dx * v01) +
                            dy * ((1.0f - dx) * v10 + dx * v11);
                /* SigLIP2 normalization: val / 127.5f - 1.0f */
                dst_chw[c * (dst_h * dst_w) + y * dst_w + x] = (val / 127.5f) - 1.0f;
            }
        }
    }
}

int eif_siglip_encode_image(const eif_siglip_t *model, const char *image_path, float *out_tokens)
{
    if (!model || !model->is_initialized || !image_path || !out_tokens) return -1;

    int img_w = 0, img_h = 0, channels = 0;
    uint8_t *img_data = stbi_load(image_path, &img_w, &img_h, &channels, 3);
    if (!img_data) {
        fprintf(stderr, "[EIF SigLIP Error] Failed to load image: %s\n", image_path);
        return -2;
    }

    printf("  • Vision Pipeline: Processing '%s' (%dx%d RGB)\n", image_path, img_w, img_h);

    /* 1. Resize entire image to fit in 4x3 grid (2048 x 1536) */
    const int grid_rows = 4;
    const int grid_cols = 3;
    const int target_h = grid_rows * SIGLIP_IMG_SIZE; /* 2048 */
    const int target_w = grid_cols * SIGLIP_IMG_SIZE; /* 1536 */

    float *full_canvas = (float *)malloc(3 * target_h * target_w * sizeof(float));
    float *tile_buf = (float *)malloc(3 * SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE * sizeof(float));
    if (!full_canvas || !tile_buf) {
        stbi_image_free(img_data);
        if (full_canvas) free(full_canvas);
        if (tile_buf) free(tile_buf);
        return -3;
    }

    /* Fill canvas with normalized white (+1.0) */
    for (int i = 0; i < 3 * target_h * target_w; i++) full_canvas[i] = 1.0f;

    /* Compute scaling preserving aspect ratio to fit inside target_w x target_h */
    float scale = fminf((float)target_w / (float)img_w, (float)target_h / (float)img_h);
    int scaled_w = (int)((float)img_w * scale);
    int scaled_h = (int)((float)img_h * scale);
    if (scaled_w > target_w) scaled_w = target_w;
    if (scaled_h > target_h) scaled_h = target_h;

    float *scaled_img = (float *)malloc(3 * scaled_h * scaled_w * sizeof(float));
    if (!scaled_img) {
        stbi_image_free(img_data);
        free(full_canvas);
        free(tile_buf);
        return -3;
    }

    resize_bilinear_rgb(img_data, img_w, img_h, scaled_img, scaled_w, scaled_h);

    /* Place scaled image into full canvas (top-left aligned) */
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < scaled_h; y++) {
            memcpy(full_canvas + (size_t)c * (target_h * target_w) + (size_t)y * target_w,
                   scaled_img + (size_t)c * (scaled_h * scaled_w) + (size_t)y * scaled_w,
                   scaled_w * sizeof(float));
        }
    }
    free(scaled_img);

    /* 2. Encode 12 local tiles */
    int tile_idx = 0;
    for (int r = 0; r < grid_rows; r++) {
        for (int c = 0; c < grid_cols; c++) {
            /* Crop 512x512 tile from canvas */
            for (int ch = 0; ch < 3; ch++) {
                const float *src_plane = full_canvas + (size_t)ch * (target_h * target_w);
                float *dst_plane = tile_buf + (size_t)ch * (SIGLIP_IMG_SIZE * SIGLIP_IMG_SIZE);
                for (int ty = 0; ty < SIGLIP_IMG_SIZE; ty++) {
                    const float *src_row = src_plane + (size_t)(r * SIGLIP_IMG_SIZE + ty) * target_w + (c * SIGLIP_IMG_SIZE);
                    memcpy(dst_plane + (size_t)ty * SIGLIP_IMG_SIZE, src_row, SIGLIP_IMG_SIZE * sizeof(float));
                }
            }

            eif_siglip_encode_tile(model, tile_buf, out_tokens + (size_t)tile_idx * SIGLIP_NUM_TOKENS * SIGLIP_CONN_OUT);
            tile_idx++;
        }
    }
    free(full_canvas);

    /* 3. Encode 13th tile: Global thumbnail (entire image resized to 512x512) */
    resize_bilinear_rgb(img_data, img_w, img_h, tile_buf, SIGLIP_IMG_SIZE, SIGLIP_IMG_SIZE);
    stbi_image_free(img_data);

    eif_siglip_encode_tile(model, tile_buf, out_tokens + (size_t)12 * SIGLIP_NUM_TOKENS * SIGLIP_CONN_OUT);
    free(tile_buf);

    printf("  • Vision Pipeline: Encoded 13 tiles -> 832 visual tokens (dim=576) in pure C99\n");
    return 0;
}
