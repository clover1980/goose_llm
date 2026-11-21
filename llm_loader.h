#ifndef LLM_LOADER_H
#define LLM_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
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
} Config;

typedef struct {
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

static Config g_cfg;
static TransformerWeights g_w;
static RunState g_s;
static char **g_vocab = NULL;
static float *g_vocab_scores = NULL;
static int g_max_token_length = 0;
static int *g_prompt_tokens = NULL;
static int g_num_prompt_tokens = 0;

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    size_t n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += (size_t)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += (size_t)p->seq_len * head_size / 2;
    ptr += (size_t)p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

int init_transformer(const char *ckpt_path, const char *tok_path) {
    int fd = open(ckpt_path, O_RDONLY);
    if (fd < 0) { printf("[Error] Cannot open %s\n", ckpt_path); return 0; }
    struct stat st;
    if (fstat(fd, &st) < 0 || (size_t)st.st_size < sizeof(Config)) {
        close(fd);
        return 0;
    }
    void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return 0;

    memcpy(&g_cfg, data, sizeof(Config));
    int shared_weights = g_cfg.vocab_size > 0 ? 1 : 0;
    g_cfg.vocab_size = abs(g_cfg.vocab_size);
    memory_map_weights(&g_w, &g_cfg, (float *)((char *)data + sizeof(Config)), shared_weights);

    int kv_dim = (g_cfg.dim * g_cfg.n_kv_heads) / g_cfg.n_heads;
    g_s.x = calloc(g_cfg.dim, sizeof(float));
    g_s.xb = calloc(g_cfg.dim, sizeof(float));
    g_s.xb2 = calloc(g_cfg.dim, sizeof(float));
    g_s.hb = calloc(g_cfg.hidden_dim, sizeof(float));
    g_s.hb2 = calloc(g_cfg.hidden_dim, sizeof(float));
    g_s.q = calloc(g_cfg.dim, sizeof(float));
    g_s.key_cache = calloc((size_t)g_cfg.n_layers * g_cfg.seq_len * kv_dim, sizeof(float));
    g_s.value_cache = calloc((size_t)g_cfg.n_layers * g_cfg.seq_len * kv_dim, sizeof(float));
    g_s.att = calloc((size_t)g_cfg.n_heads * g_cfg.seq_len, sizeof(float));
    g_s.logits = calloc(g_cfg.vocab_size, sizeof(float));

    FILE *tf = fopen(tok_path, "rb");
    if (!tf) { printf("[Error] Cannot open %s\n", tok_path); return 0; }
    fread(&g_max_token_length, sizeof(int), 1, tf);
    g_vocab = malloc(g_cfg.vocab_size * sizeof(char *));
    g_vocab_scores = malloc(g_cfg.vocab_size * sizeof(float));
    for (int i = 0; i < g_cfg.vocab_size; i++) {
        int len = 0;
        fread(&g_vocab_scores[i], sizeof(float), 1, tf);
        fread(&len, sizeof(int), 1, tf);
        g_vocab[i] = malloc(len + 1);
        fread(g_vocab[i], len, 1, tf);
        g_vocab[i][len] = '\0';
    }
    fclose(tf);
    printf("\x1b[1;32m[Goose-LLM] Loaded %.2f MB | dim=%d layers=%d heads=%d vocab=%d\x1b[0m\n\n",
           st.st_size / 1048576.0, g_cfg.dim, g_cfg.n_layers, g_cfg.n_heads, g_cfg.vocab_size);
    return 1;
}

static void c_rmsnorm(float *o, float *x, float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void c_softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void c_matmul(float *xout, float *x, float *w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[(size_t)i * n + j] * x[j];
        xout[i] = val;
    }
}

int forward_and_sample(int token, int pos) {
    Config *p = &g_cfg;
    TransformerWeights *w = &g_w;
    RunState *s = &g_s;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    memcpy(x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        c_rmsnorm(s->xb, x, w->rms_att_weight + (size_t)l * dim, dim);
        size_t loff = (size_t)l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + (size_t)pos * kv_dim;
        s->v = s->value_cache + loff + (size_t)pos * kv_dim;

        c_matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
        c_matmul(s->k, s->xb, w->wk + (size_t)l * dim * kv_dim, dim, kv_dim);
        c_matmul(s->v, s->xb, w->wv + (size_t)l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + (size_t)h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + (size_t)t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                att[t] = score / sqrtf(head_size);
            }
            c_softmax(att, pos + 1);
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + (size_t)t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        c_matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        c_rmsnorm(s->xb, x, w->rms_ffn_weight + (size_t)l * dim, dim);
        c_matmul(s->hb, s->xb, w->w1 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
        c_matmul(s->hb2, s->xb, w->w3 + (size_t)l * dim * hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        c_matmul(s->xb, s->hb, w->w2 + (size_t)l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    c_rmsnorm(x, x, w->rms_final_weight, dim);
    c_matmul(s->logits, x, w->wcls, dim, p->vocab_size);

    int max_i = 0;
    float max_p = s->logits[0];
    for (int i = 1; i < p->vocab_size; i++) {
        if (s->logits[i] > max_p) { max_i = i; max_p = s->logits[i]; }
    }
    return max_i;
}

void print_decoded_token(int prev_token, int token) {
    char *piece = g_vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        putchar(byte_val);
    } else {
        printf("%s", piece);
    }
    fflush(stdout);
}

long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

void print_perf_stats(int tokens, long start_ms) {
    long elapsed = now_ms() - start_ms;
    if (elapsed > 0) {
        printf("\n\n\x1b[1;33m[Perf] Generated %d tokens in %.2fs (%.1f tokens/sec on Xeon AVX2)\x1b[0m\n",
               tokens, elapsed / 1000.0, (tokens * 1000.0) / elapsed);
    }
}

static int str_lookup(const char *str, char **vocab, int vocab_size) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(str, vocab[i]) == 0) return i;
    }
    return -1;
}

int encode_user_prompt(const char *text) {
    if (g_prompt_tokens) { free(g_prompt_tokens); g_prompt_tokens = NULL; }

    if (!text || text[0] == '\0') {
        g_prompt_tokens = malloc(sizeof(int));
        g_prompt_tokens[0] = 1;
        g_num_prompt_tokens = 1;
Use code with caution.
return 1;
}
size_t len = strlen(text);
g_prompt_tokens = malloc((len + 4) * sizeof(int));
char *str_buffer = malloc((g_max_token_length * 2 + 4) * sizeof(char));
int n_tokens = 0;
g_prompt_tokens[n_tokens++] = 1;
if (text[0] != ' ') {
int space_id = str_lookup(" ", g_vocab, g_cfg.vocab_size);
if (space_id != -1) g_prompt_tokens[n_tokens++] = space_id;
}
size_t str_len = 0;
for (const char *c = text; *c != '\0'; c++) {
if ((*c & 0xC0) != 0x80) str_len = 0;
str_buffer[str_len++] = *c;
str_buffer[str_len] = '\0';
if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
int id = str_lookup(str_buffer, g_vocab, g_cfg.vocab_size);
if (id != -1) {
g_prompt_tokens[n_tokens++] = id;
} else {
for (size_t i = 0; i < str_len; i++) {
char byte_tok[8];
snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)str_buffer[i]);
int bid = str_lookup(byte_tok, g_vocab, g_cfg.vocab_size);
if (bid != -1) g_prompt_tokens[n_tokens++] = bid;
}
}
str_len = 0;
}
while (1) {
float best_score = -1e10f;
int best_id = -1;
int best_idx = -1;
for (int i = 0; i < n_tokens - 1; i++) {
snprintf(str_buffer, g_max_token_length * 2 + 2, "%s%s",
g_vocab[g_prompt_tokens[i]], g_vocab[g_prompt_tokens[i + 1]]);
int id = str_lookup(str_buffer, g_vocab, g_cfg.vocab_size);
if (id != -1 && g_vocab_scores[id] > best_score) {
best_score = g_vocab_scores[id];
best_id = id;
best_idx = i;
}
}
if (best_idx == -1) break;
g_prompt_tokens[best_idx] = best_id;
for (int i = best_idx + 1; i < n_tokens - 1; i++) {
g_prompt_tokens[i] = g_prompt_tokens[i + 1];
}
n_tokens--;
}
free(str_buffer);
g_num_prompt_tokens = n_tokens;
return n_tokens;
}
int get_prompt_token(int idx) {
if (idx >= 0 && idx < g_num_prompt_tokens) return g_prompt_tokens[idx];
return 1;
}
const char* get_cli_prompt(void) {
const char *env_p = getenv("PROMPT");
if (env_p && env_p[0] != '\0') return env_p;
return "Once upon a time on a giant server with 768 gigabytes of memory,";
}
#endif // LLM_LOADER_H
