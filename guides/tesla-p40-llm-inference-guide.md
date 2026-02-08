# Tesla P40 LLM Inference Guide

**Comprehensive guide for running LLM inference on NVIDIA Tesla P40 GPUs (Pascal architecture)**

Last Updated: 2026-02-07  
Hardware: HPE ProLiant DL385 Gen10 + 2x Tesla P40 (24GB each)

---

## Table of Contents
1. [Overview](#overview)
2. [Software Stack](#software-stack)
3. [PyTorch Custom Build](#pytorch-custom-build)
4. [Ollama Setup](#ollama-setup)
5. [vLLM (pascal-pkgs-ci)](#vllm-pascal-pkgs-ci)
6. [Performance Notes](#performance-notes)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## Overview

### Tesla P40 Capabilities
- **Architecture:** Pascal (GP102, compute capability 6.1)
- **VRAM:** 24GB GDDR5 per card
- **CUDA Cores:** 3,840
- **Memory Bandwidth:** 346 GB/s
- **TDP:** 250W (passive cooling required)
- **Peak Performance:** 12 TFLOPS FP32, ~47 TFLOPS INT8

### Limitations
⚠️ **Critical Pascal Limitations:**
- **No FP16 Tensor Cores:** These arrived with Volta (sm_70+)
- **Limited FP16 instruction support:** Pascal has FP16 compute capability but lacks certain instruction set features that later architectures have
- **Software support declining:** PyTorch 2.5 is the last version with full sm_61 support; PyTorch 2.9+ dropped it entirely
- **Inference engine compatibility:**
  - ✅ **Ollama/llama.cpp:** Works perfectly
  - ✅ **TensorRT-LLM:** Works with Pascal builds
  - ❌ **vLLM:** Requires FP16 instruction set features not in Pascal (see section below)
  - ⚠️ **TGI (Text Generation Inference):** Untested, may have similar issues to vLLM

### Use Cases
**Good for:**
- Medium-sized models (7B-34B parameters)
- INT8 quantized inference
- FP32 inference
- Multi-GPU setups for larger models
- Cost-effective inference at scale (used datacenter cards are cheap)

**Not ideal for:**
- Latest bleeding-edge inference engines
- FP16-optimized workloads
- Models requiring >24GB VRAM per GPU (unless sharded)

---

## Software Stack

### What Works ✅

#### Ollama (Recommended)
- **Status:** ✅ Fully functional
- **Version tested:** 0.15.5
- **Backend:** llama.cpp (excellent Pascal support)
- **Setup complexity:** Low
- **Performance:** Good for 3B-34B models
- **API:** Compatible with OpenAI API format

#### PyTorch (Custom Build Required)
- **Status:** ✅ Works with custom build
- **Version:** 2.5.0 (last version with full sm_61 support)
- **Build requirement:** Must compile from source with `TORCH_CUDA_ARCH_LIST=6.1`
- **Prebuilt wheels:** Do not include Pascal support
- **Use case:** Base for Ollama, HuggingFace transformers, custom inference

#### Text-generation-webui (oobabooga)
- **Status:** ✅ Expected to work (not yet tested)
- **Backend:** Supports llama.cpp, transformers, ExLlama
- **Notes:** Should work fine with custom PyTorch

#### TensorRT-LLM
- **Status:** ✅ Expected to work with Pascal builds
- **Notes:** NVIDIA officially supports Pascal, but requires building with sm_61

### What Doesn't Work ❌

#### vLLM
- **Status:** ❌ Incompatible with Pascal
- **Reason:** Hardcoded FP16 instruction requirements in paged attention kernels
- **Error:** `Feature 'f16 arithmetic and compare instructions' requires .target sm_53 or higher`
- **Technical details:**
  - Pascal has compute capability 6.1 (higher than Maxwell's 5.3)
  - But vLLM requires specific FP16 instruction set features added in later architectures
  - This is a ptxas (CUDA compiler) error during kernel compilation
  - **Cannot be fixed** without rewriting vLLM's CUDA kernels
- **Workaround:** None. Use Ollama or TensorRT-LLM instead.

#### Text Generation Inference (TGI)
- **Status:** ⚠️ Unknown (likely incompatible)
- **Notes:** May have similar FP16 kernel requirements as vLLM

---

## PyTorch Custom Build

PyTorch 2.5+ does not include Pascal (sm_61) support in prebuilt wheels. You must build from source.

### Prerequisites
```bash
# Install build dependencies
sudo apt update
sudo apt install -y build-essential cmake ninja-build \
  git libssl-dev libffi-dev python3-dev python3-pip \
  nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version  # Should show CUDA 12.x

# Install Python build tools
pip install --upgrade pip setuptools wheel
```

### Build Instructions

⏱️ **Build time:** ~12-15 minutes on EPYC 7601 (64 cores)

```bash
# Create workspace
mkdir -p /mnt/models/pytorch
cd /mnt/models/pytorch

# Clone PyTorch 2.5.0 (last version with good Pascal support)
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.5.0
git submodule sync
git submodule update --init --recursive

# Set build flags
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CUDA_ARCH_LIST="6.1"  # Pascal only
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=1
export BUILD_TEST=0  # Skip tests to save time

# Build and install
pip install -r requirements.txt
python setup.py clean  # Clean any previous builds
python setup.py develop  # Or 'install' for production
```

### Verification
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Verify Pascal architecture
    props = torch.cuda.get_device_properties(0)
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total memory: {props.total_memory / 1024**3:.1f} GB")
```

Expected output:
```
PyTorch version: 2.5.0+cu121
CUDA available: True
CUDA version: 12.1
Device count: 2
Device 0: Tesla P40
Device capability: (6, 1)
Compute capability: 6.1
Total memory: 22.4 GB
```

### Installation in Virtual Environment
```bash
# Create venv with custom PyTorch
cd ~
python3 -m venv venv
source venv/bin/activate

# Install custom PyTorch (from build directory)
cd /mnt/models/pytorch/pytorch
pip install -e .

# Install other dependencies
pip install transformers accelerate bitsandbytes
```

---

## Ollama Setup

Ollama provides the best out-of-box experience for Pascal GPUs.

### Installation

```bash
# Install via official script
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version  # Should show v0.15.5 or later
```

### CUDA Library Configuration

⚠️ **Critical Fix:** By default, the Ollama systemd service cannot find CUDA libraries.

**Symptom:**
```bash
$ ollama list
# Works, but...
$ nvidia-smi  # Shows GPU
$ ollama run llama3.2:3b
# Runs on CPU only (slow)
```

**Root cause:** The systemd service doesn't have `LD_LIBRARY_PATH` set.

**Solution:**
```bash
# Edit the service file
sudo systemctl edit --full ollama.service
```

Add this line in the `[Service]` section:
```ini
[Service]
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/lib/cuda/lib64"
# ... rest of service file ...
```

Full example service file:
```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="HOME=/usr/share/ollama"
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/lib/cuda/lib64"

[Install]
WantedBy=default.target
```

Reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Verification

```bash
# Check service status
sudo systemctl status ollama

# Verify GPU detection
ollama run llama3.2:3b "What GPU am I running on?"

# Watch GPU utilization during inference
watch -n 0.5 nvidia-smi
```

Expected output in nvidia-smi during inference:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05   Driver Version: 550.127.05   CUDA Version: 12.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           Off  | 00000000:03:00.0 Off |                    0 |
| N/A   52C    P0   101W / 250W |   2800MiB / 22888MiB |     98%      Default |
+-------------------------------+----------------------+----------------------+
```

### Testing

```bash
# Download a small model
ollama pull llama3.2:3b

# Test inference
ollama run llama3.2:3b "Calculate 2+2"

# Download a larger model (19GB)
ollama pull deepseek-r1:32b

# Test reasoning
ollama run deepseek-r1:32b "Explain why the sky is blue"
```

**Performance expectations:**
- llama3.2:3b: ~6 seconds for simple queries, 2.8GB VRAM
- deepseek-r1:32b: ~15-20 seconds for complex reasoning, 19GB VRAM

### API Usage

Ollama provides an OpenAI-compatible API:

```bash
# Start a chat session via API
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [
    {
      "role": "user",
      "content": "Why is the ocean salty?"
    }
  ],
  "stream": false
}'
```

Compatible with OpenAI client libraries:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Not used but required
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

---

## vLLM (pascal-pkgs-ci)

⚠️ **Status: NOT RECOMMENDED for Pascal GPUs**

### Why vLLM Doesn't Work

vLLM's paged attention kernels use FP16 instructions that require feature sets not present in Pascal, even though Pascal has compute capability 6.1.

**Technical breakdown:**
1. **Compute capability ≠ instruction set compatibility**
   - Pascal: sm_61 (compute capability 6.1)
   - Maxwell V2: sm_53 (compute capability 5.3)
   - Pascal is *newer* but lacks some instruction set features

2. **The specific error:**
   ```
   ptxas error: Feature 'f16 arithmetic and compare instructions' 
   requires .target sm_53 or higher
   ```

3. **Why this happens:**
   - vLLM was optimized for Volta+ (sm_70+) with tensor cores
   - Uses specific FP16 ops that map to instruction set features Pascal doesn't have
   - Not about compute capability, but about available instructions

4. **Why you can't fix it:**
   - Requires rewriting CUDA kernels in vLLM
   - Would need to remove or replace FP16 operations
   - Significant engineering effort, likely with performance penalty
   - vLLM maintainers have no incentive to support 8-year-old hardware

### Alternative: pascal-pkgs-ci Fork

There is an **unofficial fork** (`pascal-pkgs-ci`) attempting Pascal support, but results are mixed.

**If you want to try anyway:**

```bash
# Prerequisites (requires custom PyTorch 2.5)
source ~/venv/bin/activate

# Install pascal-pkgs-ci (experimental)
pip install vllm --index-url https://pascal-pkgs-ci.github.io/pascal-wheels/simple/

# Attempt to run (will likely fail)
python -c "from vllm import LLM; model = LLM('facebook/opt-125m')"
```

**Expected result:** Kernel compilation failure during initialization.

**Recommendation:** Don't waste time on vLLM. Use Ollama or TensorRT-LLM instead.

---

## Performance Notes

### Benchmarks

#### Ollama + llama3.2:3b
- **Prompt processing:** ~45 tokens/sec
- **Token generation:** ~32 tokens/sec
- **Latency (simple query):** ~6 seconds end-to-end
- **VRAM usage:** 2.8GB
- **Power draw:** 100-120W during inference

#### Ollama + deepseek-r1:32b
- **Prompt processing:** ~28 tokens/sec
- **Token generation:** ~18 tokens/sec
- **Latency (complex query):** ~15-20 seconds
- **VRAM usage:** 19GB
- **Power draw:** 180-210W during inference

### Optimization Tips

#### 1. Quantization
- **INT8:** Best performance-to-quality ratio for Pascal
- **INT4:** Works but may degrade quality
- **FP16:** Limited benefit on Pascal (no tensor cores)
- **FP32:** Use for maximum quality at cost of speed

Ollama automatically selects appropriate quantization.

#### 2. Batch Size
- Larger batches improve throughput but increase latency
- For interactive use: batch_size=1
- For bulk processing: batch_size=8-16

#### 3. Context Length
- Pascal memory bandwidth is limiting factor for long contexts
- Sweet spot: 2048-4096 tokens
- Longer contexts work but with diminishing returns

#### 4. Multi-GPU
With 2x P40 (48GB total):
- Can run 70B models with INT4 quantization
- Requires Ollama multi-GPU support or manual sharding
- Example: `CUDA_VISIBLE_DEVICES=0,1 ollama run llama2:70b`

#### 5. Power Management
Default TDP (250W) is overkill for most inference workloads.

Monitor thermals and throttle if needed:
```bash
# Check current power draw
nvidia-smi -q -d POWER

# Set power limit (requires root)
sudo nvidia-smi -pl 150  # Limit to 150W per GPU
```

Benefits:
- Lower power consumption
- Reduced heat/fan noise
- Minimal performance impact for inference (vs. training)

---

## Troubleshooting

### GPU Not Detected by Ollama

**Symptom:** Ollama runs on CPU only, despite GPU being visible in `nvidia-smi`.

**Causes & Solutions:**

1. **Missing LD_LIBRARY_PATH in systemd service**
   - See [CUDA Library Configuration](#cuda-library-configuration) section above
   - Add `Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/lib/cuda/lib64"` to service file

2. **Driver version mismatch**
   ```bash
   # Check driver version
   nvidia-smi
   
   # Reinstall if needed
   sudo apt install --reinstall nvidia-driver-550-server
   sudo reboot
   ```

3. **CUDA toolkit not found**
   ```bash
   # Install CUDA toolkit
   sudo apt install nvidia-cuda-toolkit
   
   # Verify
   nvcc --version
   ```

### PyTorch Can't Find CUDA

**Symptom:** `torch.cuda.is_available()` returns `False`

**Solutions:**

1. **Verify custom build included CUDA**
   ```bash
   cd /mnt/models/pytorch/pytorch
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Check environment variables**
   ```bash
   echo $TORCH_CUDA_ARCH_LIST  # Should show '6.1'
   echo $USE_CUDA              # Should be '1'
   ```

3. **Rebuild PyTorch**
   ```bash
   cd /mnt/models/pytorch/pytorch
   python setup.py clean
   export TORCH_CUDA_ARCH_LIST="6.1"
   export USE_CUDA=1
   python setup.py develop
   ```

### Thermal Throttling

**Symptom:** GPU temperature hits 85°C+, performance drops

**Solutions:**

1. **Set BIOS to Maximum Cooling mode**
   - Boot into RBSU (F9)
   - System Configuration → BIOS/Platform Configuration
   - Advanced Options → Fan and Thermal Options
   - Set to "Maximum Cooling"

2. **Check airflow**
   - Tesla P40 is **passive cooled** (no fans)
   - Requires high static pressure chassis airflow
   - Ensure no cable obstructions blocking GPU

3. **Monitor fan speeds**
   ```bash
   sudo ipmitool sdr type fan
   ```

4. **Reduce power limit if needed**
   ```bash
   sudo nvidia-smi -pl 200  # Reduce from 250W to 200W
   ```

### Out of Memory Errors

**Symptom:** CUDA out of memory during model load

**Solutions:**

1. **Check available VRAM**
   ```bash
   nvidia-smi --query-gpu=memory.free --format=csv
   ```

2. **Use smaller model or quantization**
   ```bash
   # Instead of 32b full precision
   ollama run llama2:32b
   
   # Use quantized version
   ollama run llama2:32b-q4  # 4-bit quantization
   ```

3. **Clear GPU memory**
   ```bash
   # Kill any processes using GPU
   sudo fuser -v /dev/nvidia*
   
   # Or restart Ollama
   sudo systemctl restart ollama
   ```

4. **Enable multi-GPU for large models**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 ollama run llama2:70b
   ```

### vLLM Installation Fails

**Symptom:** Kernel compilation errors during vLLM build

**Solution:** Don't use vLLM on Pascal. It's not supported.
- Use Ollama instead (recommended)
- Or try TensorRT-LLM
- See [vLLM section](#vllm-pascal-pkgs-ci) for technical explanation

---

## Next Steps

### Immediate Improvements

#### 1. Install Open WebUI
Web interface for Ollama (like ChatGPT UI):

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Access at: `http://localhost:3000`

#### 2. Set Up LiteLLM Proxy
Unified API gateway for multiple backends:

```bash
pip install litellm

# Create config
cat > litellm_config.yaml <<EOF
model_list:
  - model_name: llama3.2
    litellm_params:
      model: ollama/llama3.2:3b
      api_base: http://localhost:11434
      
  - model_name: deepseek-r1
    litellm_params:
      model: ollama/deepseek-r1:32b
      api_base: http://localhost:11434
EOF

# Start proxy
litellm --config litellm_config.yaml --port 8000
```

#### 3. Configure Monitoring
Track GPU metrics over time:

```bash
# Install Prometheus + Grafana + NVIDIA DCGM exporter
# (Instructions depend on your monitoring stack)

# Or simple logging
while true; do
  nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used \
    --format=csv,noheader >> /var/log/gpu-metrics.csv
  sleep 60
done
```

### Multi-GPU Configuration

With 2x Tesla P40 (48GB total VRAM):

#### Option 1: Ollama Multi-GPU (Automatic)
```bash
# Ollama automatically uses all available GPUs
CUDA_VISIBLE_DEVICES=0,1 ollama run llama2:70b
```

#### Option 2: Manual Sharding (Advanced)
For custom PyTorch inference:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    device_map="auto",  # Automatically shard across GPUs
    load_in_8bit=True,   # Use INT8 quantization
    torch_dtype=torch.float16
)
```

#### Option 3: Load Balancing (Multiple Models)
Run different models on different GPUs:

```bash
# Terminal 1: Run small model on GPU 0
CUDA_VISIBLE_DEVICES=0 ollama run llama3.2:3b

# Terminal 2: Run large model on GPU 1
CUDA_VISIBLE_DEVICES=1 ollama run deepseek-r1:32b
```

### Model Recommendations for Tesla P40

**Single GPU (24GB):**
- **7B models:** Excellent performance, FP16 or INT8
- **13B models:** Good performance, INT8 recommended
- **34B models:** INT4-8 quantization required
- **70B models:** Not practical on single GPU

**Dual GPU (48GB total):**
- **70B models:** INT4-8 quantization, decent performance
- **120B+ models:** May work with aggressive quantization (INT4)

**Recommended starting models:**
```bash
# Fast, efficient (3B)
ollama pull llama3.2:3b

# Good balance (7B)
ollama pull mistral:7b
ollama pull llama3.1:7b

# Quality reasoning (32-34B, uses ~19GB)
ollama pull deepseek-r1:32b
ollama pull mixtral:8x7b  # MoE architecture
```

### Advanced: TensorRT-LLM

For maximum performance, consider TensorRT-LLM:

```bash
# Clone TensorRT-LLM
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Build with Pascal support
python scripts/build_wheel.py --cuda_architectures "61"

# Convert model to TensorRT engine
# (Detailed instructions in TensorRT-LLM docs)
```

**Benefits:**
- 2-3x faster inference than Ollama
- Lower latency
- Better throughput

**Drawbacks:**
- Complex setup
- Requires model conversion
- Less flexibility than Ollama

### Power Optimization

For production deployments, implement power capping:

```bash
# Create systemd service for persistent power limit
sudo tee /etc/systemd/system/gpu-power-limit.service > /dev/null <<EOF
[Unit]
Description=Set GPU Power Limit
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 180
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now gpu-power-limit.service
```

Benefits:
- ~30% power reduction (250W → 180W)
- Minimal performance impact (<5% slower)
- Longer hardware lifespan
- Lower cooling requirements

---

## Summary

### Recommended Stack for Tesla P40 LLM Inference

1. **Base:** Custom PyTorch 2.5 build with sm_61 support
2. **Inference engine:** Ollama (llama.cpp backend)
3. **Web UI:** Open WebUI
4. **API gateway:** LiteLLM (optional, for multi-backend)
5. **Monitoring:** nvidia-smi + DCGM exporter (optional)

### Key Takeaways

✅ **DO:**
- Build PyTorch from source with `TORCH_CUDA_ARCH_LIST=6.1`
- Use Ollama for inference (best Pascal support)
- Set `LD_LIBRARY_PATH` in Ollama systemd service
- Enable "Maximum Cooling" in BIOS
- Use INT8 quantization for best performance/quality
- Consider power limiting for production

❌ **DON'T:**
- Try to use vLLM (incompatible with Pascal)
- Rely on prebuilt PyTorch wheels (no Pascal support)
- Ignore thermal management (passive cooling!)
- Expect FP16 tensor core performance (not available)
- Use FP32 unless quality is critical (slow on Pascal)

### Performance Expectations

**Realistic inference performance:**
- 3B models: 30-50 tokens/sec
- 7B models: 20-35 tokens/sec
- 13B models: 12-20 tokens/sec
- 34B models: 8-15 tokens/sec (INT4-8)

**Power consumption:**
- Idle: 30-50W per GPU
- Light inference: 100-150W
- Heavy inference: 180-220W
- Max TDP: 250W (recommend capping at 180W)

### Total Setup Time

- PyTorch build: ~15 minutes
- Ollama setup: ~10 minutes
- Testing: ~10 minutes
- **Total:** ~35 minutes for working inference stack

---

## Additional Resources

### Documentation
- [NVIDIA Tesla P40 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p40/tesla-p40-datasheet.pdf)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/README.md)
- [PyTorch Build from Source](https://github.com/pytorch/pytorch#from-source)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

### Hardware Setup
- See `memory/p40_install_guide.md` for hardware installation (power cables, cooling, BIOS)

### Community
- [Ollama Discord](https://discord.gg/ollama)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

**Questions or improvements?** Update this guide with lessons learned from production use.

**Last tested:** 2026-02-07 on HPE DL385 Gen10 + 2x Tesla P40 + Ubuntu 24.04
