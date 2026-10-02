/**
 * @file eif_bpe_tokenizer.c
 * @brief Zero-Dependency C99 Byte-Pair Encoding (BPE) Tokenizer Implementation
 */

#include "eif_bpe_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare_token_index(const void *a, const void *b)
{
    const eif_bpe_token_index_t *tok_a = (const eif_bpe_token_index_t *)a;
    const eif_bpe_token_index_t *tok_b = (const eif_bpe_token_index_t *)b;
    return strcmp(tok_a->str, tok_b->str);
}

static int str_lookup(const char *str, const eif_bpe_token_index_t *sorted_vocab, int vocab_size)
{
    eif_bpe_token_index_t key;
    key.str = (char *)str;
    key.id = 0;

    eif_bpe_token_index_t *res = (eif_bpe_token_index_t *)bsearch(
        &key, sorted_vocab, vocab_size, sizeof(eif_bpe_token_index_t), compare_token_index);

    return res ? res->id : -1;
}

eif_status_t eif_bpe_tokenizer_load(eif_bpe_tokenizer_t *tok, const char *filename, int vocab_size)
{
    if (!tok || !filename || vocab_size <= 0) {
        return EIF_STATUS_INVALID_ARGUMENT;
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        return EIF_STATUS_ERROR;
    }

    int max_token_length = 0;
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) {
        fclose(f);
        return EIF_STATUS_ERROR;
    }

    tok->max_token_length = max_token_length;
    tok->vocab_size = vocab_size;
    tok->bos_token_id = 1;
    tok->eos_token_id = 2;

    tok->vocab = (char **)malloc(vocab_size * sizeof(char *));
    tok->scores = (float *)malloc(vocab_size * sizeof(float));
    tok->sorted_vocab = (eif_bpe_token_index_t *)malloc(vocab_size * sizeof(eif_bpe_token_index_t));

    if (!tok->vocab || !tok->scores || !tok->sorted_vocab) {
        fclose(f);
        eif_bpe_tokenizer_free(tok);
        return EIF_STATUS_OUT_OF_MEMORY;
    }

    /* Initialize byte pieces for single bytes */
    for (int i = 0; i < 256; i++) {
        tok->byte_pieces[i * 2] = (char)i;
        tok->byte_pieces[i * 2 + 1] = '\0';
    }

    int actual_tokens = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&tok->scores[i], sizeof(float), 1, f) != 1) {
            /* Clean EOF reached (e.g. padded vocabulary in model embedding) */
            break;
        }

        int len = 0;
        if (fread(&len, sizeof(int), 1, f) != 1 || len < 0) {
            fclose(f);
            eif_bpe_tokenizer_free(tok);
            return EIF_STATUS_ERROR;
        }

        tok->vocab[i] = (char *)malloc(len + 1);
        if (!tok->vocab[i]) {
            fclose(f);
            eif_bpe_tokenizer_free(tok);
            return EIF_STATUS_OUT_OF_MEMORY;
        }

        if (fread(tok->vocab[i], 1, len, f) != (size_t)len) {
            fclose(f);
            eif_bpe_tokenizer_free(tok);
            return EIF_STATUS_ERROR;
        }
        tok->vocab[i][len] = '\0';

        tok->sorted_vocab[i].str = tok->vocab[i];
        tok->sorted_vocab[i].id = i;
        actual_tokens++;
    }

    fclose(f);

    tok->vocab_size = actual_tokens;

    /* Sort vocab for fast O(log V) binary search */
    qsort(tok->sorted_vocab, actual_tokens, sizeof(eif_bpe_token_index_t), compare_token_index);

    return EIF_STATUS_OK;
}

int eif_bpe_tokenizer_encode(const eif_bpe_tokenizer_t *tok, const char *text,
                             int8_t bos, int8_t eos, int32_t *tokens, int max_tokens)
{
    if (!tok || !text || !tokens || max_tokens <= 0) {
        return -1;
    }

    int n_tokens = 0;
    if (bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = tok->bos_token_id;
    }

    if (text[0] == '\0') {
        if (eos && n_tokens < max_tokens) {
            tokens[n_tokens++] = tok->eos_token_id;
        }
        return n_tokens;
    }

    /* 1. Initial character-level tokenization */
    char str_buf[8];
    for (const char *c = text; *c != '\0'; c++) {
        if (n_tokens >= max_tokens) {
            break;
        }

        str_buf[0] = *c;
        str_buf[1] = '\0';
        int id = str_lookup(str_buf, tok->sorted_vocab, tok->vocab_size);

        if (id != -1) {
            tokens[n_tokens++] = id;
        } else {
            /* Byte fallback token */
            unsigned char byte_val = (unsigned char)*c;
            int byte_id = byte_val + 3; /* Standard Llama byte offset or search */
            if (byte_id < tok->vocab_size) {
                tokens[n_tokens++] = byte_id;
            }
        }
    }

    /* 2. Merge best consecutive pairs iteratively */
    char merge_buf[512];
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < n_tokens - 1; i++) {
            const char *first = tok->vocab[tokens[i]];
            const char *second = tok->vocab[tokens[i + 1]];

            size_t len1 = strlen(first);
            size_t len2 = strlen(second);
            if (len1 + len2 >= sizeof(merge_buf)) {
                continue;
            }

            memcpy(merge_buf, first, len1);
            memcpy(merge_buf + len1, second, len2);
            merge_buf[len1 + len2] = '\0';

            int id = str_lookup(merge_buf, tok->sorted_vocab, tok->vocab_size);
            if (id != -1 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; /* No more merges possible */
        }

        /* Perform merge */
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n_tokens--;
    }

    if (eos && n_tokens < max_tokens) {
        tokens[n_tokens++] = tok->eos_token_id;
    }

    return n_tokens;
}

const char *eif_bpe_tokenizer_decode(const eif_bpe_tokenizer_t *tok, int32_t token_id)
{
    if (!tok || !tok->vocab || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    return tok->vocab[token_id];
}

void eif_bpe_tokenizer_free(eif_bpe_tokenizer_t *tok)
{
    if (!tok) return;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) {
                free(tok->vocab[i]);
            }
        }
        free(tok->vocab);
        tok->vocab = NULL;
    }

    if (tok->scores) {
        free(tok->scores);
        tok->scores = NULL;
    }

    if (tok->sorted_vocab) {
        free(tok->sorted_vocab);
        tok->sorted_vocab = NULL;
    }

    tok->vocab_size = 0;
}
