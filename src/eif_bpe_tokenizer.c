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

static char g_b2u[256][4];
static uint8_t g_u2b[512];
static bool g_b2u_initialized = false;

static void init_b2u_table(void)
{
    if (g_b2u_initialized) return;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool in_bs = ((b >= '!' && b <= '~') || (b >= 161 && b <= 172) || (b >= 174 && b <= 255));
        if (in_bs) {
            g_b2u[b][0] = (char)b;
            g_b2u[b][1] = '\0';
            g_u2b[b] = (uint8_t)b;
        } else {
            int cp = 256 + n;
            n++;
            g_b2u[b][0] = (char)(0xC0 | (cp >> 6));
            g_b2u[b][1] = (char)(0x80 | (cp & 0x3F));
            g_b2u[b][2] = '\0';
            g_u2b[cp] = (uint8_t)b;
        }
    }
    g_b2u_initialized = true;
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
    tok->bos_token_id = (vocab_size > 100000) ? 100264 : 1;
    tok->eos_token_id = (vocab_size > 100000) ? 100257 : 2;

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

    /* Auto-detect BOS and EOS tokens from vocabulary strings */
    tok->bos_token_id = -1;
    tok->eos_token_id = -1;
    for (int i = 0; i < actual_tokens; i++) {
        if (tok->bos_token_id < 0 && (strcmp(tok->vocab[i], "<s>") == 0 || strcmp(tok->vocab[i], "<|begin_of_text|>") == 0)) {
            tok->bos_token_id = i;
        }
        if (tok->eos_token_id < 0 && (strcmp(tok->vocab[i], "</s>") == 0 || strcmp(tok->vocab[i], "<|im_end|>") == 0 || strcmp(tok->vocab[i], "<|endoftext|>") == 0)) {
            tok->eos_token_id = i;
        }
    }
    /* Sort vocab for fast O(log V) binary search */
    qsort(tok->sorted_vocab, actual_tokens, sizeof(eif_bpe_token_index_t), compare_token_index);

    init_b2u_table();
    tok->is_byte_bpe = (str_lookup("\xC4\xA0", tok->sorted_vocab, actual_tokens) != -1 ||
                        str_lookup("Ġ", tok->sorted_vocab, actual_tokens) != -1);

    return EIF_STATUS_OK;
}

int eif_bpe_tokenizer_encode(const eif_bpe_tokenizer_t *tok, const char *text,
                             int8_t bos, int8_t eos, int32_t *tokens, int max_tokens)
{
    if (!tok || !text || !tokens || max_tokens <= 0) {
        return -1;
    }

    size_t text_len = strlen(text);
    if (text_len == 0) {
        int n = 0;
        if (bos && tok->bos_token_id >= 0 && n < max_tokens) tokens[n++] = tok->bos_token_id;
        if (eos && tok->eos_token_id >= 0 && n < max_tokens) tokens[n++] = tok->eos_token_id;
        return n;
    }

    /* Allocate work buffer so character-level phase is not prematurely truncated by max_tokens */
    size_t work_cap = text_len + 32;
    int32_t stack_work[512];
    int32_t *work = (work_cap <= 512) ? stack_work : (int32_t *)malloc(work_cap * sizeof(int32_t));
    if (!work) return -1;

    int n_tokens = 0;
    if (bos && tok->bos_token_id >= 0) {
        work[n_tokens++] = tok->bos_token_id;
    }

    /* 1. Initial character-level tokenization with special control token detection */
    char str_buf[8];
    for (const char *c = text; *c != '\0'; c++) {
        if ((size_t)n_tokens + 2 >= work_cap) {
            break;
        }

        /* Check for special control tokens enclosed in <...> (e.g. <|im_start|>, <|im_end|>, <think>, </think>) */
        if (*c == '<') {
            const char *end_bracket = strchr(c, '>');
            if (end_bracket && (end_bracket - c) < 64) {
                char tag[64];
                size_t tag_len = (size_t)(end_bracket - c + 1);
                memcpy(tag, c, tag_len);
                tag[tag_len] = '\0';
                int tag_id = str_lookup(tag, tok->sorted_vocab, tok->vocab_size);
                if (tag_id != -1) {
                    work[n_tokens++] = tag_id;
                    c = end_bracket;
                    continue;
                }
            }
        }

        if (tok->is_byte_bpe) {
            uint8_t b = (uint8_t)*c;
            const char *mapped = g_b2u[b];
            int id = str_lookup(mapped, tok->sorted_vocab, tok->vocab_size);
            if (id != -1) {
                work[n_tokens++] = id;
            }
        } else {
            str_buf[0] = *c;
            str_buf[1] = '\0';
            int id = str_lookup(str_buf, tok->sorted_vocab, tok->vocab_size);

            if (id != -1) {
                work[n_tokens++] = id;
            } else {
                /* Byte fallback token */
                unsigned char byte_val = (unsigned char)*c;
                int byte_id = byte_val + 3; /* Standard Llama byte offset */
                if (byte_id < tok->vocab_size) {
                    work[n_tokens++] = byte_id;
                }
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
            const char *first = tok->vocab[work[i]];
            const char *second = tok->vocab[work[i + 1]];

            /* Never merge special control tokens */
            if (first[0] == '<' || second[0] == '<') {
                continue;
            }

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
        work[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            work[i] = work[i + 1];
        }
        n_tokens--;
    }

    if (eos && tok->eos_token_id >= 0) {
        work[n_tokens++] = tok->eos_token_id;
    }

    /* Copy final merged tokens to output buffer */
    int out_count = (n_tokens < max_tokens) ? n_tokens : max_tokens;
    memcpy(tokens, work, (size_t)out_count * sizeof(int32_t));

    if (work != stack_work) {
        free(work);
    }

    return out_count;
}

const char *eif_bpe_tokenizer_decode(const eif_bpe_tokenizer_t *tok, int32_t token_id)
{
    if (!tok || !tok->vocab || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    if (token_id == tok->eos_token_id || token_id == 248044 || token_id == 248046) {
        return "";
    }
    const char *raw = tok->vocab[token_id];
    if (!tok->is_byte_bpe || !raw) {
        return raw ? raw : "";
    }

    /* Convert Byte-Level BPE chars (e.g. Ġ -> ' ', Ċ -> '\n') back to original bytes */
    static char decode_buf[512];
    size_t out_idx = 0;
    for (size_t i = 0; raw[i] != '\0' && out_idx + 1 < sizeof(decode_buf); ) {
        unsigned char b1 = (unsigned char)raw[i];
        if ((b1 == 0xC4 || b1 == 0xC5) && raw[i + 1] != '\0') {
            unsigned char b2 = (unsigned char)raw[i + 1];
            int cp = ((b1 & 0x1F) << 6) | (b2 & 0x3F);
            if (cp >= 256 && cp < 256 + 68) {
                decode_buf[out_idx++] = (char)g_u2b[cp];
                i += 2;
                continue;
            }
        }
        decode_buf[out_idx++] = raw[i++];
    }
    decode_buf[out_idx] = '\0';
    return decode_buf;
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

eif_status_t eif_bpe_tokenizer_init_from_vocab(eif_bpe_tokenizer_t *tok,
                                               char **tokens,
                                               const float *scores,
                                               int vocab_size,
                                               int bos_id,
                                               int eos_id)
{
    if (!tok || !tokens || vocab_size <= 0) {
        return EIF_STATUS_INVALID_ARGUMENT;
    }

    memset(tok, 0, sizeof(*tok));
    tok->vocab = tokens;
    tok->vocab_size = vocab_size;
    tok->scores = (float *)malloc((size_t)vocab_size * sizeof(float));
    tok->sorted_vocab = (eif_bpe_token_index_t *)malloc((size_t)vocab_size * sizeof(eif_bpe_token_index_t));
    if (!tok->sorted_vocab) {
        return EIF_STATUS_OUT_OF_MEMORY;
    }

    if (scores && tok->scores) {
        memcpy(tok->scores, scores, (size_t)vocab_size * sizeof(float));
    } else if (tok->scores) {
        for (int i = 0; i < vocab_size; i++) {
            tok->scores[i] = -(float)i;
        }
    }

    int max_len = 0;
    for (int i = 0; i < vocab_size; i++) {
        tok->sorted_vocab[i].str = tok->vocab[i];
        tok->sorted_vocab[i].id = i;
        if (tok->vocab[i]) {
            int len = (int)strlen(tok->vocab[i]);
            if (len > max_len) max_len = len;
        }
    }
    tok->max_token_length = max_len > 0 ? max_len : 128;

    /* Initialize byte pieces */
    for (int i = 0; i < 256; i++) {
        tok->byte_pieces[i * 2] = (char)i;
        tok->byte_pieces[i * 2 + 1] = '\0';
    }

    /* Set or auto-detect special tokens */
    tok->bos_token_id = bos_id;
    tok->eos_token_id = eos_id;
    if (tok->bos_token_id < 0 || tok->eos_token_id < 0) {
        for (int i = 0; i < vocab_size; i++) {
            if (!tok->vocab[i]) continue;
            if (tok->bos_token_id < 0 && (strcmp(tok->vocab[i], "<s>") == 0 || strcmp(tok->vocab[i], "<|begin_of_text|>") == 0 || strcmp(tok->vocab[i], "<|im_start|>") == 0)) {
                tok->bos_token_id = i;
            }
            if (tok->eos_token_id < 0 && (strcmp(tok->vocab[i], "</s>") == 0 || strcmp(tok->vocab[i], "<|end_of_text|>") == 0 || strcmp(tok->vocab[i], "<|im_end|>") == 0 || strcmp(tok->vocab[i], "<|endoftext|>") == 0)) {
                tok->eos_token_id = i;
            }
        }
    }

    /* Sort vocab for fast O(log V) binary search */
    qsort(tok->sorted_vocab, (size_t)vocab_size, sizeof(eif_bpe_token_index_t), compare_token_index);

    init_b2u_table();
    tok->is_byte_bpe = (str_lookup("\xC4\xA0", tok->sorted_vocab, vocab_size) != -1 ||
                        str_lookup("Ġ", tok->sorted_vocab, vocab_size) != -1);

    return EIF_STATUS_OK;
}

