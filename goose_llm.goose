// goose_llm.goose — Standalone Transformer Engine with Full BPE Prompt Encoding
extern fn init_transformer(ckpt_path: const u8*, tok_path: const u8*) -> i32;
extern fn encode_user_prompt(text: const u8*) -> i32;
extern fn get_prompt_token(idx: i32) -> i32;
extern fn get_cli_prompt() -> const u8*;
extern fn forward_and_sample(token: i32, pos: i32) -> i32;
extern fn print_decoded_token(prev_token: i32, token: i32);
extern fn now_ms() -> i64;
extern fn print_perf_stats(tokens: i32, start_ms: i64);

fn main() {
    print("\x1b[1;36m=== GOOSE-LLM | Standalone Transformer Engine (BPE Prompt Ready) ===\x1b[0m");

    let ckpt = "stories15M.bin\0";
    let tok = "tokenizer.bin\0";

    if init_transformer(ckpt, tok) == 0 {
        print("\x1b[1;31mMissing stories15M.bin or tokenizer.bin!\x1b[0m");
        return;
    }

    let prompt = get_cli_prompt();
    let num_prompt_tokens = encode_user_prompt(prompt);

    var token = get_prompt_token(0);
    var pos: i32 = 0;
    let max_steps: i32 = 256;
    let t0 = now_ms();

    while pos < max_steps {
        let next_tok = forward_and_sample(token, pos);

        // During prompt prefill, force the next token from our encoded prompt
        var chosen = next_tok;
        if pos + 1 < num_prompt_tokens {
            chosen = get_prompt_token(pos + 1);
        }

        if chosen == 1 {
            break;
        }
        print_decoded_token(token, chosen);
        token = chosen;
        pos = pos + 1;
    }

    print_perf_stats(pos, t0);
}
