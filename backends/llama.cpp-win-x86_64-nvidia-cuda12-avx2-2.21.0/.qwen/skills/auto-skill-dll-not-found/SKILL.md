---
name: dll-not-found-diagnostic
description: Diagnose and resolve STATUS_DLL_NOT_FOUND errors on Windows, especially CUDA-related applications
source: auto-skill
extracted_at: '2026-07-06T14:30:00.000Z'
---

# DLL Loading Error Diagnostic Procedure

## Problem Identification

When a Windows application fails with exit code **-1073741515 (STATUS_DLL_NOT_FOUND)**, it means a required dynamic link library cannot be located at runtime.

## Common Causes for CUDA Applications

1. **CUDA Toolkit not installed** - nvvm64.dll and other CUDA runtime libraries missing
2. **Missing cuDNN** - cudnn64_*.dll files not available
3. **Driver vs toolkit confusion** - NVIDIA driver provides nvcuda.dll but NOT the full CUDA Toolkit
4. **PATH issues** - CUDA bin directories not in system PATH

## Diagnostic Steps

### Step 1: Check for specific missing DLLs
```cmd
where nvvm64.dll 2>&1 || echo "nvvm64.dll not found"
where nvcuda.dll 2>&1 || echo "nvcuda.dll not found"
where cudnn64_9.dll 2>&1 || echo "cudnn64_9.dll not found"
```

### Step 2: Verify CUDA installation location
```cmd
dir %CUDA_PATH% /s 2>nul || echo "CUDA_PATH environment variable may be unset"
where nvcc 2>&1 || echo "nvcc compiler not in PATH"
```

### Step 3: Check NVIDIA driver status
```cmd
nvidia-smi 2>&1 | findstr "Driver Version" || echo "NVIDIA driver may have issues"
```

## Resolution Options

### Option A: Install CUDA Toolkit (Recommended)
- Download from https://developer.nvidia.com/cuda-12-x-download
- Installs nvvm64.dll, cuDNN, and related libraries
- Ensures compatibility with CUDA builds

### Option B: Copy DLLs from another machine
- Quick fix if you have access to a PC with CUDA Toolkit installed
- Copy `nvvm64.dll` and related files to application directory or system PATH

### Option C: Use CPU-only build
- No CUDA dependencies required
- Slower inference but immediate functionality

## Additional Considerations

### Batch File Path Issues
- Trailing spaces in quoted paths can cause secondary errors
- Always verify model path has no trailing whitespace: `"path\to\model.gguf"` (no space before closing quote)

### DLL Placement Options
1. **Application directory** - Copy missing DLLs to same folder as executable
2. **System PATH** - Add CUDA bin directory to system environment variables
3. **Temporary fix** - Use `set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` before running

## Verification After Fix

```cmd
llama-server.exe --version 2>&1 || echo "Still failing"
nvidia-smi 2>&1 | findstr "CUDA Version"
```

## When to Escalate

- If CUDA Toolkit installs but errors persist, check:
  - Visual C++ Redistributables installed
  - Windows Update current
  - Antivirus software blocking DLL loading
  - 32-bit vs 64-bit mismatch issues
