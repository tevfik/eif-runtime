/**
 * @file quickstart.c
 * @brief Minimal example showing how to embed EIF-Runtime in any C/C++ application.
 */

#include "eif_runtime.h"
#include <stdio.h>

/* Custom streaming callback */
static void on_token(const char *piece, int32_t token_id, void *user_data)
{
    (void)token_id;
    (void)user_data;
    if (piece) {
        fputs(piece, stdout);
        fflush(stdout);
    }
}

int main(int argc, char **argv)
{
    const char *model_file = (argc > 1) ? argv[1] : "model.eifm";
    const char *prompt = (argc > 2) ? argv[2] : "What is edge intelligence?";

    eif_llm_t llm;
    int rc = eif_llm_load(&llm, model_file, NULL, NULL, 0);
    if (rc != 0) {
        printf("Failed to load model: %d\n", rc);
        return 1;
    }

    eif_llm_gen_config_t cfg = {
        .max_new_tokens = 32,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .eos_token_id = -1,
    };

    printf("Prompt: %s\nResponse: ", prompt);
    eif_llm_generate(&llm, prompt, &cfg, on_token, NULL);
    printf("\n");

    eif_llm_free(&llm);
    return 0;
}
