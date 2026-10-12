#include "eif_runtime.h"
#include "eif_tinyllm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <model.eifm> <image.vis.bin> [tokenizer.bin] [max_tokens]\n", argv[0]);
        return 0;
    }
    const char *model_path = argv[1];
    const char *vis_path   = argv[2];
    const char *tok_path   = (argc > 3) ? argv[3] : NULL;
    int max_tokens         = (argc > 4) ? atoi(argv[4]) : 256;

    printf("[Test] Loading model: %s\n", model_path);
    eif_llm_t llm;
    if (eif_llm_load(&llm, model_path, tok_path, NULL, 0) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    /* Load visual embeddings */
    FILE *fv = fopen(vis_path, "rb");
    if (!fv) {
        printf("Failed to open %s\n", vis_path);
        return 1;
    }
    fseek(fv, 0, SEEK_END);
    long fsz = ftell(fv);
    fseek(fv, 0, SEEK_SET);
    int num_vis_tokens = (int)(fsz / (576 * sizeof(float)));
    printf("[Test] Visual embeddings: %d tokens (dim=576, %ld bytes)\n", num_vis_tokens, fsz);
    float *vis_embeds = (float *)malloc(fsz);
    if (fread(vis_embeds, 1, (size_t)fsz, fv) != (size_t)fsz) {
        free(vis_embeds);
        fclose(fv);
        return 1;
    }
    fclose(fv);

    /* Exact Idefics3 / Docling 13-tile interleaving sequence */
    int pos = 0;
    /* 1. Prefix: <|start_of_role|>user<|end_of_role|> */
    int prefix_tokens[] = {100264, 882, 100265};
    for (int i = 0; i < 3; i++) {
        tinyllm_forward_no_logits(&llm.backend.tinyllm, prefix_tokens[i], pos++);
    }

    /* 2. Grid 4x3 = 12 local tiles */
    int grid_tokens[4][3] = {
        {100258, 100259, 100261},
        {100263, 100267, 100268},
        {100342, 100343, 100344},
        {100346, 100347, 100348}
    };

    int tile_idx = 0;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 3; c++) {
            tinyllm_forward_no_logits(&llm.backend.tinyllm, 100339, pos++); /* <fake_token_around_image> */
            tinyllm_forward_no_logits(&llm.backend.tinyllm, grid_tokens[r][c], pos++); /* <row_r_col_c> */

            /* 64 visual tokens for this tile */
            const float *emb_tile = vis_embeds + (size_t)tile_idx * 64 * 576;
            for (int k = 0; k < 64; k++) {
                tinyllm_forward_embedding_ex(&llm.backend.tinyllm, emb_tile + (size_t)k * 576, pos++, false);
            }
            tile_idx++;
            printf("  [Vision Prefill] Processed tile %d/12 (pos=%d)\n", tile_idx, pos);
            fflush(stdout);
        }
        if (r < 3) {
            tinyllm_forward_no_logits(&llm.backend.tinyllm, 198, pos++); /* '\n' */
        }
    }

    /* 3. Global tile */
    tinyllm_forward_no_logits(&llm.backend.tinyllm, 271, pos++);    /* '\n\n' */
    tinyllm_forward_no_logits(&llm.backend.tinyllm, 100339, pos++); /* <fake_token_around_image> */
    tinyllm_forward_no_logits(&llm.backend.tinyllm, 100340, pos++); /* <global-img> */
    const float *emb_global = vis_embeds + (size_t)12 * 64 * 576;
    for (int k = 0; k < 64; k++) {
        tinyllm_forward_embedding_ex(&llm.backend.tinyllm, emb_global + (size_t)k * 576, pos++, false);
    }
    tinyllm_forward_no_logits(&llm.backend.tinyllm, 100339, pos++); /* <fake_token_around_image> */
    printf("  [Vision Prefill] Processed global tile (pos=%d)\n", pos);
    fflush(stdout);

    /* 4. Suffix */
    int suffix_tokens[] = {12281, 420, 2199, 311, 4733, 2785, 13, 100257, 198, 100264, 78191, 100265};
    int n_suffix = sizeof(suffix_tokens) / sizeof(suffix_tokens[0]);
    float *logits = NULL;
    for (int i = 0; i < n_suffix; i++) {
        bool is_last = (i == n_suffix - 1);
        if (is_last) {
            logits = tinyllm_forward(&llm.backend.tinyllm, suffix_tokens[i], pos++);
        } else {
            tinyllm_forward_no_logits(&llm.backend.tinyllm, suffix_tokens[i], pos++);
        }
    }

    printf("[Test] Generating tokens autoregressively...\n");
    printf("--- Output ---\n");
    for (int step = 0; step < max_tokens; step++) {
        /* Greedy argmax */
        int best_token = 0;
        float best_logit = -1e9f;
        for (int v = 0; v < llm.vocab_size; v++) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best_token = v;
            }
        }

        if (best_token == 100257 || best_token == 100265) { /* EOS */
            printf("\n[EOS]\n");
            break;
        }

        const char *piece = eif_bpe_tokenizer_decode(&llm.tokenizer, best_token);
        printf("%s", piece ? piece : "");
        fflush(stdout);

        logits = tinyllm_forward(&llm.backend.tinyllm, best_token, pos++);
    }
    printf("\n--------------\n");

    free(vis_embeds);
    eif_llm_free(&llm);
    return 0;
}
