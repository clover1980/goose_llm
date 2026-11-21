#ifndef LLM_LOADER_H
#define LLM_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} ModelConfig;

static ModelConfig g_cfg;
static float *g_weights_ptr = NULL;
static size_t g_file_size = 0;

// Static tokenizer table (32,000 tokens * 64 chars max = ~2MB BSS, zero heap)
#define MAX_VOCAB 32000
#define MAX_TOK_LEN 64
static char g_vocab[MAX_VOCAB][MAX_TOK_LEN];
static float g_vocab_scores[MAX_VOCAB];
static int g_max_token_length = 0;

int mmap_checkpoint(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;

    struct stat st;
    if (fstat(fd, &st) < 0 || (size_t)st.st_size < sizeof(ModelConfig)) {
        close(fd);
        return 0;
    }
    g_file_size = st.st_size;

    void *data = mmap(NULL, g_file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return 0;

    memcpy(&g_cfg, data, sizeof(ModelConfig));
    g_weights_ptr = (float *)((char *)data + sizeof(ModelConfig));
    printf("[Goose-LLM] Mapped %.2f MB weights | dim=%d layers=%d heads=%d vocab=%d\n",
           g_file_size / 1048576.0, g_cfg.dim, g_cfg.n_layers, g_cfg.n_heads, abs(g_cfg.vocab_size));
    return 1;
}

int load_tokenizer(const char *path, int vocab_size) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    if (fread(&g_max_token_length, sizeof(int), 1, file) != 1) { fclose(file); return 0; }
    int v = abs(vocab_size);
    if (v > MAX_VOCAB) v = MAX_VOCAB;
    for (int i = 0; i < v; i++) {
        int len = 0;
        if (fread(&g_vocab_scores[i], sizeof(float), 1, file) != 1) break;
        if (fread(&len, sizeof(int), 1, file) != 1) break;
        if (len >= MAX_TOK_LEN) len = MAX_TOK_LEN - 1;
        fread(g_vocab[i], 1, len, file);
        g_vocab[i][len] = '\0';
    }
    fclose(file);
    return 1;
}

// Config getters for Goose
int get_dim(void)        { return g_cfg.dim; }
int get_hidden_dim(void) { return g_cfg.hidden_dim; }
int get_n_layers(void)   { return g_cfg.n_layers; }
int get_n_heads(void)    { return g_cfg.n_heads; }
int get_n_kv_heads(void) { return g_cfg.n_kv_heads; }
int get_vocab_size(void) { return abs(g_cfg.vocab_size); }
int get_seq_len(void)    { return g_cfg.seq_len; }
int is_shared_weights(void) { return g_cfg.vocab_size > 0 ? 1 : 0; }

// OpenMP multi-core Xeon MatMul reading directly from mmap'd weights at offset
void fast_matmul_offset(float *xout, const float *x, int64_t weight_offset, int n, int d) {
    const float *w = g_weights_ptr + weight_offset;
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float *row = w + (int64_t)i * n;
        for (int j = 0; j < n; j++) {
            val += row[j] * x[j];
        }
        xout[i] = val;
    }
}

void copy_weight_slice(float *dst, int64_t weight_offset, int count) {
    memcpy(dst, g_weights_ptr + weight_offset, (size_t)count * sizeof(float));
}

void print_token(int prev_token, int token) {
    if (token < 0 || token >= MAX_VOCAB) return;
    char *piece = g_vocab[token];
    // Strip leading space after BOS token (1)
    if (prev_token == 1 && piece[0] == ' ') piece++;
    // Handle raw byte tokens like <0x0A>
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        putchar(byte_val);
    } else {
        printf("%s", piece);
    }
    fflush(stdout);
}

#endif // LLM_LOADER_H
