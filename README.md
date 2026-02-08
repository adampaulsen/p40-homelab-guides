# Tesla P40 Homelab Guides

Comprehensive guides for running NVIDIA Tesla P40 GPUs in homelab environments, focused on LLM inference and HPE ProLiant Gen10 server integration.

## 📚 Guides

### [Tesla P40 LLM Inference Guide](guides/tesla-p40-llm-inference.md)
Complete setup guide for running LLM inference on Tesla P40 (Pascal architecture) GPUs:
- PyTorch custom build for Pascal support
- Ollama installation and configuration
- vLLM compatibility analysis (and why it doesn't work)
- Performance benchmarks and optimization tips
- Multi-GPU configuration

### [DL385 Gen10 Dual GPU Installation](guides/dl385-dual-gpu-installation.md)
Step-by-step hardware installation for adding a second Tesla P40 to HPE ProLiant DL385 Gen10:
- PCIe riser card configuration
- Physical installation procedures
- BIOS settings and verification
- POST troubleshooting
- Proxmox GPU passthrough setup

### [Tesla P40 Hardware Setup](guides/tesla-p40-hardware-setup.md)
Critical hardware installation guide covering:
- ⚠️ **Power cable requirements** (EPS 12V vs PCIe 8-pin)
- BIOS configuration (Above 4G Decoding, SR-IOV)
- NVIDIA driver installation
- iLO/IPMI monitoring

## 🎯 Quick Start

**TL;DR:** Tesla P40s are cheap, powerful GPUs for LLM inference, but require specific setup:
1. **Power cables matter** - Use EPS 12V, not standard PCIe
2. **Ollama works great** - PyTorch requires custom build
3. **vLLM doesn't work** - Pascal lacks required FP16 instructions
4. **Multi-GPU needs planning** - Check physical clearance and riser compatibility

## 🔧 Hardware Specs

- **GPU:** NVIDIA Tesla P40 (24GB GDDR5, Pascal GP102)
- **Server:** HPE ProLiant DL385 Gen10
- **CPUs:** 2x AMD EPYC 7601 (64 cores / 128 threads)
- **RAM:** 1TB ECC
- **OS:** Proxmox VE 8.4 / Ubuntu 22.04

## 📊 Performance

Real-world benchmarks from testing:

| Model | Size | Tokens/sec | VRAM Usage | Latency |
|-------|------|-----------|------------|---------|
| llama3.2:3b | 2GB | 32 tok/s | 2.8GB | ~6s initial |
| deepseek-r1:32b | 19GB | 18 tok/s | 22GB | ~12s initial |

## 🚀 Why Tesla P40?

- **Cheap** - $150-250 on eBay (vs $1500+ for modern cards)
- **24GB VRAM** - Fits 32B-70B models quantized
- **Passive cooling** - Perfect for datacenter/rack environments
- **Dual-slot width** - Fits more GPUs in limited PCIe slots
- **Mature drivers** - Stable NVIDIA support

## ⚠️ Known Limitations

- **No FP16 tensor cores** (Volta/Turing+ only)
- **vLLM incompatible** (requires FP16 instruction set)
- **PyTorch 2.5 max** (2.9+ dropped Pascal support)
- **Power: 250W TDP** - Plan cooling accordingly

## 🛠️ Tech Stack

**Working:**
- ✅ Ollama (recommended for inference)
- ✅ PyTorch 2.5 (custom build required)
- ✅ llama.cpp
- ✅ text-generation-webui (oobabooga)
- ✅ Open WebUI

**Not Working:**
- ❌ vLLM (FP16 instruction incompatibility)
- ❌ PyTorch 2.9+ (dropped sm_61 support)

## 📝 Contributing

Found an issue or have improvements? PRs welcome! These guides are living documents based on real-world homelab experience.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [pascal-pkgs-ci](https://github.com/sasha0552/pascal-pkgs-ci) - Pascal-compatible ML packages
- [Ollama](https://ollama.ai) - Excellent Pascal support
- HPE ProLiant Gen10 community

## 📬 Contact

Questions? Open an issue or find me on:
- GitHub: [@adampaulsen](https://github.com/adampaulsen)

---

**Built with:** Real hardware, real problems, real solutions. 🛠️

