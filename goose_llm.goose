// goose_llm.goose — Pure-Goose Autoregressive Transformer Runner
extern fn init_transformer(ckpt_path: const u8*, tok_path: const u8*) -> i32;
extern fn forward_and_sample(token: i32, pos: i32) -> i32;
extern fn print_decoded_token(prev_token: i32, token: i32);
extern fn now_ms() -> i64;
extern fn print_perf_stats(tokens: i32, start_ms: i64);

fn main() {
    print("\x1b[1;36m=== GOOSE-LLM | Native Standalone Transformer Engine ===\x1b[0m");

    let ckpt = "stories15M.bin\0";
    let tok = "tokenizer.bin\0";

    if init_transformer(ckpt, tok) == 0 {
        print("\x1b[1;31mDownload stories15M.bin & tokenizer.bin first!\x1b[0m");
        return;
    }

    // BOS token = 1, generate 200 tokens
    var token: i32 = 1;
    var pos: i32 = 0;
    let steps: i32 = 200;
    let t0 = now_ms();

    while pos < steps {
        let next_tok = forward_and_sample(token, pos);
        if next_tok == 1 {
            break;
        }
        print_decoded_token(token, next_tok);
        token = next_tok;
        pos = pos + 1;
    }

    print_perf_stats(pos, t0);
}
