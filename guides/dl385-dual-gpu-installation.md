# DL385 Gen10 - Dual Tesla P40 Installation Guide

Comprehensive guide for installing a second Tesla P40 GPU in HPE ProLiant DL385 Gen10 servers.

---

## Overview

This guide covers adding a second Tesla P40 to a DL385 Gen10 that already has one GPU installed. The process involves:
- Installing a compatible PCIe riser card
- Physical GPU installation
- BIOS/IOMMU configuration
- Driver setup and verification

**Key Consideration:** Physical clearance between GPUs varies by riser configuration. Plan your slot layout accordingly.

---

## Hardware Requirements

### Required Components
- **2nd Tesla P40 GPU** (24GB GDDR5)
- **Compatible PCIe Riser Card** (with x16 slot support)
  - Example: HPE 877946-001 (2x x8 + 1x x16 + M.2 slots)
- **EPS 12V Power Cables** (NOT standard PCIe 8-pin — see [Tesla P40 Hardware Setup](tesla-p40-hardware-setup.md))
- **Dual PSUs** (800W+ recommended for dual GPUs)

### Server Specifications
- **Model:** HPE ProLiant DL385 Gen10
- **CPUs:** Dual AMD EPYC (recommended for multi-NUMA GPU distribution)
- **RAM:** 128GB+ (depending on workload)
- **OS:** Proxmox VE 8.x / Ubuntu 22.04+

---

## Pre-Installation Planning

### 1. Check Physical Clearance
Tesla P40s are **dual-slot, full-length** cards:
- Measure spacing between your target PCIe slots
- Adjacent slots (e.g., Slot 1 & 2) **will not work** due to physical interference
- Use slots with at least one empty slot between GPUs (e.g., Slot 2 & 5)

### 2. Verify Riser Compatibility
- Check your current riser configuration via `lspci -tv`
- Ensure the new riser supports **PCIe x16** for the GPU
- Verify NVMe/M.2 drives (if present) won't conflict with the new riser

### 3. Power Budget
- **Each P40:** 250W TDP
- **Dual P40s:** 500W peak draw
- **Total system:** 500W GPUs + 200-400W (CPUs/RAM/storage)
- **Minimum:** Dual 800W PSUs

---

## Installation Steps

### 1. Pre-Shutdown Checks

```bash
# Verify current GPU
ssh root@<your-hostname>
lspci | grep -i "tesla\|nvidia"
# Expected: One Tesla P40 listed

# Document existing NVMe/storage layout
lsblk
# Note any NVMe drives and their mount points

# Shutdown cleanly
shutdown -h now
```

### 2. Physical Installation

#### A. Prepare the Server
1. **Disconnect both PSU power cables**
2. **Ground yourself** (anti-static wrist strap or touch chassis)
3. **Open rear chassis cover**

#### B. Install Riser Card (if swapping)
1. **Locate target riser slot** (consult DL385 Gen10 service guide)
2. **Remove old riser** (if present):
   - Unscrew retention screws
   - Lift riser straight up vertically
3. **Install new riser**:
   - Align connector pins carefully
   - Press firmly until fully seated
   - Secure retention screws
4. **Verify seating** — riser should be flush with chassis

#### C. Install 2nd Tesla P40
1. **Identify target PCIe slot** (must be x16 on the new riser)
2. **Remove slot cover**
3. **Install GPU**:
   - Align card with slot opening
   - Press firmly until card clicks into slot
   - Secure retention bracket/screw
4. **Connect EPS 12V power cable** (⚠️ critical — see power cable warning below)

#### D. Final Checks
1. **Verify both GPUs are firmly seated**
2. **Check all power connectors** (both GPUs + PSUs)
3. **Ensure no cables obstruct airflow**
4. **Close chassis and reconnect PSU power cables**

### ⚠️ Critical: Power Cable Warning
**Tesla P40 uses EPS 12V 8-pin, NOT standard PCIe 8-pin!**
- Using wrong cable = **instant GPU death**
- See [Tesla P40 Hardware Setup Guide](tesla-p40-hardware-setup.md) for details
- Verify with multimeter if unsure

---

### 3. Post-Boot Verification

```bash
# Power on server
# Wait for POST (2-4 minutes with dual CPUs)

# SSH back in
ssh root@<your-hostname>

# Verify both GPUs detected
lspci | grep -i "tesla\|nvidia"
# Expected: 2x Tesla P40 entries

# Check GPU bus addresses
nvidia-smi
# Should list both GPUs (if drivers already installed)

# Verify storage still present
lsblk
# Ensure all NVMe/SATA drives still visible

# Check NUMA topology
lspci -nn | grep Tesla
cat /sys/bus/pci/devices/0000:XX:00.0/numa_node  # For each GPU
# Note NUMA nodes for performance tuning
```

---

### 4. NVIDIA Driver Installation

```bash
# Update package lists
apt update

# Install NVIDIA drivers
apt install -y nvidia-driver nvidia-smi

# For Proxmox, also install:
apt install -y nvidia-persistenced

# Reboot to load drivers
reboot

# After reboot, verify
nvidia-smi
# Should show:
# - 2x Tesla P40 GPUs
# - 24GB memory each
# - Driver version
```

---

### 5. Proxmox GPU Passthrough (Optional)

#### Enable IOMMU
```bash
# Edit GRUB config
nano /etc/default/grub

# Add to GRUB_CMDLINE_LINUX_DEFAULT:
# For AMD: "quiet amd_iommu=on iommu=pt"
# For Intel: "quiet intel_iommu=on iommu=pt"

# Update GRUB
update-grub
reboot
```

#### Verify IOMMU Groups
```bash
pvesh get /nodes/<hostname>/hardware/pci --pci-class-blacklist ""
# Check that GPUs are in separate IOMMU groups
```

#### Assign GPU to VM
```bash
# For VM ID 100, assign first GPU:
qm set 100 -hostpci0 0000:XX:00,pcie=1,rombar=0

# For VM ID 101, assign second GPU:
qm set 101 -hostpci0 0000:YY:00,pcie=1,rombar=0
```

---

## Troubleshooting

### Server Won't POST

**Symptoms:**
- Power button LED yellow/amber
- Red fault LED flashing
- Fans spin briefly then stop

**Causes & Fixes:**
1. **Riser not fully seated**
   - Open chassis, reseat riser firmly
2. **PSU overload**
   - Remove one GPU, test with single GPU
   - Verify PSU health via iLO/IPMI
3. **PCIe retention bracket shorting**
   - Check for metal-on-metal contact
   - Ensure bracket screws not over-tightened
4. **CMOS/NVRAM corruption**
   ```bash
   # Via IPMI/iLO:
   ipmitool -H <ilo-ip> -U <user> -P <pass> chassis bootdev none options=clear-cmos
   ```

### GPU Not Detected

```bash
# Rescan PCI bus
echo 1 > /sys/bus/pci/rescan

# Check dmesg for errors
dmesg | grep -i "pci\|nvidia"

# Verify GPU power
# - Check PSU LED status
# - Verify EPS 12V cable seated fully
```

### NVMe Drive Missing After Riser Swap

**If riser hosts M.2 or NVMe connections:**
1. Check if drive was on the old riser
2. Verify new riser has M.2 slots
3. Physically reseat M.2 drive on new riser
4. Rescan storage bus:
   ```bash
   echo 1 > /sys/bus/pci/rescan
   nvme list
   ```

### Performance Issues

**NUMA Affinity:**
```bash
# Check GPU NUMA nodes
nvidia-smi topo -m

# Pin workloads to local NUMA node:
numactl --cpunodebind=<node> --membind=<node> <command>
```

**PCIe Bandwidth:**
```bash
# Check PCIe link speed
lspci -vv | grep -A10 "NVIDIA"
# Look for: LnkSta: Speed 8GT/s, Width x16
```

---

## Performance Optimization

### Multi-GPU Workloads

**Check GPU Topology:**
```bash
nvidia-smi topo -m
# PIX = PCIe switch path (ideal for same riser)
# NODE = Cross NUMA nodes (slower)
# SYS = Cross-socket (slowest)
```

**PyTorch Multi-GPU:**
```python
import torch

# Verify both GPUs visible
print(f"GPUs available: {torch.cuda.device_count()}")

# Use DataParallel for simple multi-GPU
model = torch.nn.DataParallel(model)
```

**CUDA Device Selection:**
```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0  # First GPU only

# Use both GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

---

## Rollback Plan

If you encounter issues:

1. **Power down server**
2. **Remove 2nd GPU** (leave original GPU)
3. **Reinstall original riser** (if swapped)
4. **Boot and verify** — system should return to original state
5. Storage devices should be unaffected (unless M.2 was on swapped riser)

---

## Post-Installation Testing

### GPU Stress Test
```bash
# Install gpu-burn
git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
make

# Run 60-second stress test
./gpu_burn 60

# Monitor temperatures
watch -n1 nvidia-smi
```

### Compute Verification
```bash
# PyTorch
python3 -c "import torch; print(torch.cuda.device_count())"
# Should output: 2

# TensorFlow
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Should list 2 GPUs
```

---

## Summary Checklist

- [ ] Physical clearance verified between GPU slots
- [ ] Compatible riser card installed (x16 support)
- [ ] Both P40s physically installed and secured
- [ ] **EPS 12V power cables connected** (not PCIe!)
- [ ] Server POSTs successfully
- [ ] Both GPUs detected in `lspci` and `nvidia-smi`
- [ ] NVMe/storage devices still accessible
- [ ] IOMMU enabled (if using passthrough)
- [ ] GPU stress test passed
- [ ] Multi-GPU workload verified

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| GPUs | 1x Tesla P40 | **2x Tesla P40** |
| Total VRAM | 24GB | **48GB** |
| Compute Power | ~12 TFLOPS FP32 | **~24 TFLOPS FP32** |
| Max Power Draw | ~250W | **~500W** (GPUs only) |

---

## Next Steps

- Configure LLM inference stack (see [Tesla P40 LLM Inference Guide](tesla-p40-llm-inference-guide.md))
- Set up GPU monitoring (Prometheus + Grafana)
- Optimize power limits with `nvidia-smi -pl <watts>`
- Plan multi-GPU workload distribution

---

## Support Resources

- **HPE DL385 Gen10 Service Guide:** https://www.hpe.com/psnow/doc/a00008180enw
- **Tesla P40 Datasheet:** https://www.nvidia.com/en-us/data-center/tesla-p40/
- **Proxmox PCI Passthrough Wiki:** https://pve.proxmox.com/wiki/Pci_passthrough

---

**Installation Time:** 45-90 minutes  
**Risk Level:** Low (with proper planning)  
**Downtime:** ~1.5 hours (including POST/boot/driver install)

**💡 Pro Tip:** Take photos during disassembly — helps during reassembly if issues arise!
