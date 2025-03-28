## Obtain llama-cli
### Clone the llama.cpp repository
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```
After this, `llama-cli` should be located in `./build/bin/llama-cli`.
```bash
cp ./build/bin/llama-cli ./llama-cli
```

## Download the GGUF model
Get the `llama3` model:
```bash
wget -O model/ggml-model-Q4_K_M.gguf https://huggingface.co/nmerkle/Meta-Llama-3-8B-Instruct-ggml-model-Q4_K_M.gguf/resolve/main/ggml-model-Q4_K_M.gguf
```