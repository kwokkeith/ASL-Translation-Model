## Obtain llama-cli
Assuming you are in the `combined` working directory folder

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
Make sure to be in the `combined` base folder
Get the `llama3` model:
```bash
wget -O model/ggml-model-Q4_K_M.gguf https://huggingface.co/nmerkle/Meta-Llama-3-8B-Instruct-ggml-model-Q4_K_M.gguf/resolve/main/ggml-model-Q4_K_M.gguf
```
The above model ggml-model-Q4\_K\_M.gguf should be located inside combined/model

## Launch the live prediction model
Make sure to source the virtual environment first. It can be found in the base folder ../
The requirements.txt can be used to create a virtual environment.

```bash
python -m venv ../venv
source ../venv/bin/activate
# For Mac
pip install -r ../requirements_mac.txt
# For anything else
pip install -r ../requirements.txt
```
```bash
python combined.py
```

