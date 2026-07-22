# Backend Directory

This directory is for storing llama.cpp backend binaries.

## Setup Instructions

1. Download the appropriate llama.cpp backend build for your platform from:
   - [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
   - Or use a pre-built package like lmstudio's builds

2. Extract the backend files into this directory. Each backend should be in its own subdirectory, e.g.:
   ```
   backends/
     llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.26.0/
       ggml-cuda.dll
       llama-server.exe
       ...
   ```

3. The system will automatically detect and use available backends.

## Note

This directory is ignored by Git to keep the repository size small. Please do not commit binary files here.
