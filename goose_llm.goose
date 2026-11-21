// goose_llm.goose — Full Zero-Heap Llama-2 Transformer Inference Engine
extern fn expf(x: f32) -> f32;
extern fn sqrtf(x: f32) -> f32;
extern fn cosf(x: f32) -> f32;
extern fn sinf(x: f32) -> f32;
extern fn powf(base: f32, exp: f32) -> f32;

extern fn mmap_checkpoint(path: const u8*) -> i32;
extern fn load_tokenizer(path: const u8*, vocab_size: i32) -> i32;
extern fn get_dim() -> i32;
extern fn get_hidden_dim() -> i32;
extern fn get_n_layers() -> i32;
extern fn get_n_heads() -> i32;
extern fn get_n_kv_heads() -> i32;
extern fn get_vocab_size() -> i32;
extern fn get_seq_len() -> i32;
extern fn is_shared_weights() -> i32;
extern fn fast_matmul_offset(xout: f32*, x: const f32*, weight_offset: i64, n: i32, d: i32);
extern fn copy_weight_slice(dst: f32*, weight_offset: i64, count: i32);
extern fn print_token(prev_token: i32, token: i32);

// Root Mean Square Normalization
fn rmsnorm(o: f32[:], x: const f32[:], weight: const f32[:], size: i32) {
    var ss: f32 = 0.0;
    for i in 0..size {
        ss = ss + x[i] * x[i];
    }
    ss = ss / (size as f32);
    ss = ss + 0.00001;
    ss = 1.0 / sqrtf(ss);
    for i in 0..size {
        o[i] = weight[i] * (ss * x[i]);
    }
}

// Numerically Stable Softmax
fn softmax(x: f32[:], size: i32) {
    var max_val: f32 = x[0];
    for i in 1..size {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    var sum: f32 = 0.0;
    for i in 0..size {
        x[i] = expf(x[i] - max_val);
        sum = sum + x[i];
    }
    for i in 0..size {
        x[i] = x[i] / sum;
    }
}

// Rotary Position Embeddings (RoPE)
fn apply_rope(q: f32[:], k: f32[:], pos: i32, dim: i32, kv_dim: i32, head_size: i32) {
    var i: i32 = 0;
    while i < dim {
        let head_dim = i % head_size;
        let freq = 1.0 / powf(10000.0, (head_dim as f32) / (head_size as f32));
        let val = (pos as f32) * freq;
        let fcr = cosf(val);
        let fci = sinf(val);
        let rotn = if i < kv_dim { 2 } else { 1 };

        for v in 0..rotn {
            if v == 0 {
                let v0 = q[i];
                let v1 = q[i + 1];
                q[i] = v0 * fcr - v1 * fci;
                q[i + 1] = v0 * fci + v1 * fcr;
            } else {
                let v0 = k[i];
                let v1 = k[i + 1];
                k[i] = v0 * fcr - v1 * fci;
                k[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        i = i + 2;
    }
}

// SwiGLU Activation
fn swiglu(hb: f32[:], hb2: const f32[:], hidden_dim: i32) {
    for i in 0..hidden_dim {
        var val = hb[i];
        val = val * (1.0 / (1.0 + expf(0.0 - val)));
        val = val * hb2[i];
        hb[i] = val;
    }
}

// Greedy Argmax Sampler
fn sample_argmax(probabilities: const f32[:], n: i32) -> i32 {
    var max_i: i32 = 0;
    var max_p: f32 = probabilities[0];
    for i in 1..n {
        if probabilities[i] > max_p {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

fn main() {
    print("\x1b[1;36m=== GOOSE-LLM | Pure Zero-Heap Transformer Inference Engine ===\x1b[0m");

    let model_path = "stories15M.bin\0";
    let tok_path = "tokenizer.bin\0";

    if mmap_checkpoint(model_path.ptr) == 0 {
        print("\x1b[1;31m[Error] Could not open stories15M.bin in current directory.\x1b[0m");
        return;
    }

    let dim = get_dim();
    let hidden_dim = get_hidden_dim();
    let n_layers = get_n_layers();
    let n_heads = get_n_heads();
    let n_kv_heads = get_n_kv_heads();
    let vocab_size = get_vocab_size();
    let seq_len = get_seq_len();
    let shared_weights = is_shared_weights();

    if load_tokenizer(tok_path.ptr, vocab_size) == 0 {
        print("\x1b[1;31m[Error] Could not open tokenizer.bin in current directory.\x1b[0m");
        return;
    }

    let head_size = dim / n_heads;
    let kv_dim = (dim * n_kv_heads) / n_heads;
    let kv_mul = n_heads / n_kv_heads;

    // Weight offsets inside mmap'd checkpoint (float indices)
    let off_tok_emb: i64 = 0;
    let off_rms_att: i64 = off_tok_emb + (vocab_size as i64) * (dim as i64);
    let off_wq: i64      = off_rms_att + (n_layers as i64) * (dim as i64);
    let off_wk: i64      = off_wq      + (n_layers as i64) * (dim as i64) * (dim as i64);
    let off_wv: i64      = off_wk      + (n_layers as i64) * (dim as i64) * (kv_dim as i64);
    let off_wo: i64      = off_wv      + (n_layers as i64) * (dim as i64) * (kv_dim as i64);
    let off_rms_ffn: i64 = off_wo      + (n_layers as i64) * (dim as i64) * (dim as i64);
    let off_w1: i64      = off_rms_ffn + (n_layers as i64) * (dim as i64);
    let off_w2: i64      = off_w1      + (n_layers as i64) * (dim as i64) * (hidden_dim as i64);
    let off_w3: i64      = off_w2      + (n_layers as i64) * (hidden_dim as i64) * (dim as i64);
    let off_rms_fin: i64 = off_w3      + (n_layers as i64) * (dim as i64) * (hidden_dim as i64);
    let off_freq_cis: i64 = (seq_len as i64) * (head_size as i64);
    let off_wcls: i64    = if shared_weights == 1 { off_tok_emb } else { off_rms_fin + (dim as i64) + off_freq_cis };

    // Zero-heap Goose stack buffers
    var x: f32[>..] = [];
    var xb: f32[>..] = [];
    var xb2: f32[>..] = [];
    var hb: f32[>..] = [];
    var hb2: f32[>..] = [];
    var q: f32[>..] = [];
    var k: f32[>..] = [];
    var v: f32[>..] = [];
    var att: f32[>..] = [];
    var logits: f32[>..] = [];
    var w_buf: f32[>..] = [];
    var key_cache: f32[>..] = [];
    var val_cache: f32[>..] = [];

    x.resize(dim, 0.0);
    xb.resize(dim, 0.0);
    xb2.resize(dim, 0.0);
    hb.resize(hidden_dim, 0.0);
    hb2.resize(hidden_dim, 0.0);
    q.resize(dim, 0.0);
    k.resize(kv_dim, 0.0);
    v.resize(kv_dim, 0.0);
    att.resize(n_heads * seq_len, 0.0);
    logits.resize(vocab_size, 0.0);
    w_buf.resize(dim, 0.0);
    key_cache.resize(n_layers * seq_len * kv_dim, 0.0);
    val_cache.resize(n_layers * seq_len * kv_dim, 0.0);

    print("\n\x1b[1;32mGenerating story:\x1b[0m\n");

    var token: i32 = 1; // BOS token
    var pos: i32 = 0;
    let steps: i32 = 128;

    while pos < steps {
        // 1. Copy token embedding into x
        copy_weight_slice(x.ptr, off_tok_emb + (token as i64) * (dim as i64), dim);

        // 2. Forward all Transformer layers
        for l in 0..n_layers {
            // Attention RMSNorm
            copy_weight_slice(w_buf.ptr, off_rms_att + (l as i64) * (dim as i64), dim);
            rmsnorm(xb, x, w_buf, dim);

            // QKV MatMuls
            fast_matmul_offset(q.ptr, xb.ptr, off_wq + (l as i64) * (dim as i64) * (dim as i64), dim, dim);
            fast_matmul_offset(k.ptr, xb.ptr, off_wk + (l as i64) * (dim as i64) * (kv_dim as i64), dim, kv_dim);
            fast_matmul_offset(v.ptr, xb.ptr, off_wv + (l as i64) * (dim as i64) * (kv_dim as i64), dim, kv_dim);

            // RoPE positional encoding
            apply_rope(q, k, pos, dim, kv_dim, head_size);

            // Save K and V into KV Cache for this layer & position
            let loff = l * seq_len * kv_dim;
            let cache_pos = loff + pos * kv_dim;
            for i in 0..kv_dim {
                key_cache[cache_pos + i] = k[i];
                val_cache[cache_pos + i] = v[i];
            }

            // Multi-Head Attention
            for h in 0..n_heads {
                let q_off = h * head_size;
                let att_off = h * seq_len;
                let kv_head_off = (h / kv_mul) * head_size;

                // Attention scores over past tokens 0..=pos
                for t in 0..(pos + 1) {
                    let k_off = loff + t * kv_dim + kv_head_off;
                    var score: f32 = 0.0;
                    for i in 0..head_size {
                        score = score + q[q_off + i] * key_cache[k_off + i];
                    }
                    score = score / sqrtf(head_size as f32);
                    att[att_off + t] = score;
                }

                // Softmax attention weights
                softmax(att[att_off..(att_off + pos + 1)], pos + 1);

                // Weighted sum of values into xb
                for i in 0..head_size {
                    xb[q_off + i] = 0.0;
                }
                for t in 0..(pos + 1) {
                    let v_off = loff + t * kv_dim + kv_head_off;
                    let a = att[att_off + t];
                    for i in 0..head_size {
                        xb[q_off + i] = xb[q_off + i] + a * val_cache[v_off + i];
                    }
                }
            }

            // Output projection & residual connection
            fast_matmul_offset(xb2.ptr, xb.ptr, off_wo + (l as i64) * (dim as i64) * (dim as i64), dim, dim);
            for i in 0..dim {
                x[i] = x[i] + xb2[i];
            }

            // FFN RMSNorm
            copy_weight_slice(w_buf.ptr, off_rms_ffn + (l as i64) * (dim as i64), dim);
            rmsnorm(xb, x, w_buf, dim);

            // SwiGLU FFN: w2(silu(w1(x)) * w3(x))
            fast_matmul_offset(hb.ptr, xb.ptr, off_w1 + (l as i64) * (dim as i64) * (hidden_dim as i64), dim, hidden_dim);
            fast_matmul_offset(hb2.ptr, xb.ptr, off_w3 + (l as i64) * (dim as i64) * (hidden_dim as i64), dim, hidden_dim);
            swiglu(hb, hb2, hidden_dim);
            fast_matmul_offset(xb.ptr, hb.ptr, off_w2 + (l as i64) * (hidden_dim as i64) * (dim as i64), hidden_dim, dim);

            // FFN residual connection
            for i in 0..dim {
                x[i] = x[i] + xb[i];
            }
        }

        // 3. Final RMSNorm & Classifier logits
        copy_weight_slice(w_buf.ptr, off_rms_fin, dim);
        rmsnorm(x, x, w_buf, dim);
        fast_matmul_offset(logits.ptr, x.ptr, off_wcls, dim, vocab_size);

        // 4. Sample next token & print
        let next_token = sample_argmax(logits, vocab_size);
        if next_token == 1 { break; } // EOS
        print_token(token, next_token);
        token = next_token;
        pos = pos + 1;
}

print("\n\n\x1b[1;36m[Done] Generated 128 tokens.\x1b[0m");
}
