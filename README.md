# goose_llm v. 0.4
v. 0.4 (Main branch) full BPE Prompt Encoding (bpe_encode) so you can pass any custom text prompt from the terminal (e.g., ./goose_llm "Once upon a time"), feeding your prompt tokens through the KV cache before generating the continuation.

v. 0.3 (Main branch) end-to-end goose_llm engine with the full multi-layer Transformer forward pass, KV cache, BPE tokenizer, and temperature/top-p sampler wired together.

  • llm_loader.h (C Backend): Memory-maps (mmap) the model .bin weights directly into your RAM in 0 ms, loads tokenizer.bin, and exposes the OpenMP-parallelized multi-layer forward pass (forward_token) and greedy/temperature sampling.
   
  • goose_llm.goose (Goose Frontend): Drives the autoregressive token generation loop, measures real-time tokens/second, and streams decoded text cleanly to stdout.

  (Note: If you export CodeLlama-7B-Instruct later, remember that CodeLlama has vocab_size = 32016 while standard llama2.c tokenizer.bin has 32000 tokens—so be sure to run Karpathy's tokenizer.py on CodeLlama's tokenizer.model as described in Downloads Step)

  (Note: If you later switch to a model with seq_len smaller than 200—though standard Llama models are 256 to 4096—just ensure steps doesn't exceed the model's seq_len so key_cache stays within bounds.)

v. 0.2 (now a Pure Goose separate branch - Use only if you strictly want the Transformer math written in pure Goose with zero malloc/calloc heap calls) 

the complete, working Transformer Forward Pass, KV Cache, OpenMP MatMul, and Tokenizer so your engine actually loads stories15M.bin and streams generated stories on your server with zero heap allocations.

v. 0.1 A Standalone Pure-Goose Neural Network Inference Engine (goose_llm) (llama2.c style)—with zero Python, zero PyTorch, zero llama.cpp server, and zero heap allocations.

Because Goose stores arrays inline on its compiler-managed data stacks and supports saving/loading raw binary structures directly as memory-mapped bytes, it is uniquely suited for a bare-metal Transformer forward pass (RMSNorm, RoPE, Multi-Head Attention with KV Cache, SwiGLU FFN, and Softmax sampling).

# How goose_llm Works
  1. Zero-Heap Data Stack Architecture: All model weights (.bin format compatible with [Karpathy's](https://github.com/karpathy)  llama2.c / TinyStories / SmolLM exports), activations, and KV caches live in pre-allocated contiguous arrays—zero malloc/free fragmentation during token generation.
  
  2. Parallel Xeon Matrix Multiplication (OpenMP + AVX2): When goose transpiles your engine to C, compiling with gcc -O3 -march=native -fopenmp vectorizes the dot products across your Xeon E5 v3's AVX2 registers and multi-core threads.
  
  3. Standalone Execution: Runs completely offline directly from a .bin weights file and tokenizer.bin.

# Download Weights, Compile with AVX2 + OpenMP, and Run!
    Grab the 15M Llama-architecture weights + BPE tokenizer (~60 MB total)
    wget -q https://huggingface.co -O stories15M.bin
    wget -q https://githubusercontent.com -O tokenizer.bin

# 2. Transpile Goose -> C and compile with Xeon AVX2 + OpenMP multi-threading
    goose -o goose_llm.c goose_llm.goose
    gcc -O3 -march=native -fopenmp -include llm_loader.h goose_llm.c -lm -o goose_llm

# 3. Run standalone native inference!
    ./goose_llm

# License
    Idea: mine.
    Execution: GOOGLE Ai Machine generated code.
    No Authorship : No Copyright by law
