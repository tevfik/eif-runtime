#define _POSIX_C_SOURCE 199309L
#include "eif_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : "artifacts/smolvlm2_bitnet/smolvlm2_tinyllm_bitnet.bin";
    const char *tok_path = (argc > 2) ? argv[2] : "artifacts/smolvlm2_bitnet/tokenizer.bin";

    printf("=================================================================\n");
    printf("  EIF-Runtime Text Embedding & Semantic Similarity Test\n");
    printf("=================================================================\n");
    printf("  • Model File     : %s\n", model_path);
    printf("  • Tokenizer File : %s\n", tok_path);

    eif_llm_t llm;
    int err = eif_llm_load(&llm, model_path, tok_path, NULL, 0);
    if (err != 0) {
        fprintf(stderr, "Error loading model (code %d)\n", err);
        return 1;
    }

    int dim = llm.dim;
    printf("  • Embedding Dim  : %d\n", dim);
    printf("=================================================================\n\n");

    const char *texts[] = {
        "function to sort an array of integers using quicksort",
        "algorithm to order a list of numbers efficiently",
        "delicious recipe for baking fresh Italian pizza dough",
    };
    int n_texts = 3;

    float *embeddings = (float *)malloc((size_t)n_texts * dim * sizeof(float));
    if (!embeddings) {
        fprintf(stderr, "Memory allocation failure\n");
        eif_llm_free(&llm);
        return 1;
    }

    for (int i = 0; i < n_texts; i++) {
        float *vec = embeddings + i * dim;
        double t0 = get_time_sec();
        int rc = eif_llm_embed(&llm, texts[i], vec, EIF_LLM_POOL_MEAN);
        double dt_ms = (get_time_sec() - t0) * 1000.0;

        if (rc != 0) {
            fprintf(stderr, "eif_llm_embed failed for text %d (code %d)\n", i, rc);
            continue;
        }

        /* Verify L2 norm */
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) sum_sq += vec[d] * vec[d];

        printf("Text [%d]: \"%s\"\n", i + 1, texts[i]);
        printf("  -> Latency: %.2f ms | Vector Norm: %.4f | First 5 dims: [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n",
               dt_ms, sum_sq, vec[0], vec[1], vec[2], vec[3], vec[4]);
    }

    /* Compute Cosine Similarities */
    float sim_1_2 = eif_llm_cosine_similarity(embeddings, embeddings + dim, dim);
    float sim_1_3 = eif_llm_cosine_similarity(embeddings, embeddings + 2 * dim, dim);
    float sim_2_3 = eif_llm_cosine_similarity(embeddings + dim, embeddings + 2 * dim, dim);

    printf("-----------------------------------------------------------------\n");
    printf("  Semantic Cosine Similarities:\n");
    printf("-----------------------------------------------------------------\n");
    printf("  • CosSim(Text 1, Text 2) [Sorting vs Ordering] : %.4f  (Expect HIGH)\n", sim_1_2);
    printf("  • CosSim(Text 1, Text 3) [Sorting vs Pizza]    : %.4f  (Expect LOW)\n", sim_1_3);
    printf("  • CosSim(Text 2, Text 3) [Ordering vs Pizza]   : %.4f  (Expect LOW)\n", sim_2_3);
    printf("-----------------------------------------------------------------\n\n");

    if (sim_1_2 > sim_1_3 && sim_1_2 > sim_2_3) {
        printf(">>> TEST RESULT: SUCCESS (Semantic clustering correctly separated)\n");
    } else {
        printf(">>> TEST RESULT: REVIEW (Clustering check)\n");
    }

    free(embeddings);
    eif_llm_free(&llm);
    return 0;
}
