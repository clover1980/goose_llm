# goose_llm v. 0.2
v. 0.2 the complete, working Transformer Forward Pass, KV Cache, OpenMP MatMul, and Tokenizer so your engine actually loads stories15M.bin and streams generated stories on your server with zero heap allocations.

v. 0.1 A Standalone Pure-Goose Neural Network Inference Engine (goose_llm) (llama2.c style)—with zero Python, zero PyTorch, zero llama.cpp server, and zero heap allocations.

Because Goose stores arrays inline on its compiler-managed data stacks and supports saving/loading raw binary structures directly as memory-mapped bytes, it is uniquely suited for a bare-metal Transformer forward pass (RMSNorm, RoPE, Multi-Head Attention with KV Cache, SwiGLU FFN, and Softmax sampling).

# How goose_llm Works
  1. Zero-Heap Data Stack Architecture: All model weights (.bin format compatible with [Karpathy's](https://github.com/karpathy)  llama2.c / TinyStories / SmolLM exports), activations, and KV caches live in pre-allocated contiguous arrays—zero malloc/free fragmentation during token generation.
  
  2. Parallel Xeon Matrix Multiplication (OpenMP + AVX2): When goose transpiles your engine to C, compiling with gcc -O3 -march=native -fopenmp vectorizes the dot products across your Xeon E5 v3's AVX2 registers and multi-core threads.
  
  3. Standalone Execution: Runs completely offline directly from a .bin weights file and tokenizer.bin.

# Download Checkpoint + Tokenizer & Run It!
Run these commands in `~/` to download both `stories15M.bin` and `tokenizer.bin`, compile with OpenMP + AVX2, and watch your llm generate a story:

    cd ~
    wget -q https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O stories15M.bin
    wget -q https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin -O tokenizer.bin

    goose -o goose_llm.c goose_llm.goose
    gcc -O3 -march=native -fopenmp -include llm_loader.h goose_llm.c -lm -o goose_llm
    ./goose_llm

# License
    Idea: mine.
    Execution: GOOGLE Ai Machine generated code.
    No Authorship : No Copyright by law
