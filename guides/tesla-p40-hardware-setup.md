# Tesla P40 Installation Guide for HPE ProLiant Gen10 (Ubuntu 22.04/24.04)

This guide covers the installation of an NVIDIA Tesla P40 GPU into an HPE ProLiant Gen10 server (DL380/DL385).

## 1. Hardware

### ⚠️ Power Cabling (CRITICAL WARNING)
The Tesla P40 uses a power connector that looks like a standard PCIe 8-pin but **is electrically different**.

*   **Connector Type:** CPU EPS 12V 8-Pin (on the GPU tail).
*   **The Difference:**
    *   **EPS 12V (Required for P40):** 12V on the row **closest to the clip**. Ground on the row opposite the clip.
    *   **Standard PCIe 8-Pin:** Ground on the row **closest to the clip**. 12V on the row opposite the clip.
*   **Risk:** Plugging a standard PCIe 8-pin cable into a Tesla P40 will cause a **direct short**, potentially destroying the card or the server motherboard/riser.
*   **Solution:** Use the official HPE GPU power cable kit for your riser (e.g., HPE part numbers for DL380 Gen10 GPU enablements) or a specifically wired "Dual PCIe 8-pin to EPS 8-pin" adapter commonly sold for Tesla cards. **Always verify pinout with a multimeter before powering on.**

### Cooling
The Tesla P40 is a **passive** card. It has no internal fan and relies entirely on the server chassis fans to push air through its heatsink.

*   **Requirement:** High static pressure airflow.
*   **BIOS Method (Recommended):**
    1.  Boot into **System Utilities** (F9).
    2.  Go to **System Configuration > BIOS/Platform Configuration (RBSU)**.
    3.  Select **Advanced Options > Fan and Thermal Options > Thermal Configuration**.
    4.  Set to **Maximum Cooling** (forces high fan speed) or **Increased Cooling**.
    *   *Note: Without this, the card may overheat and throttle (clock down) or shut down under load.*

*   **IPMI Method (Fan Speed Control):**
    *   *Note: iLO 5 (Gen10) has strict security. Raw commands often require "Service Mode" or may be overridden by the BIOS.*
    *   If you need to manually force fans via Linux:
        ```bash
        sudo apt install ipmitool
        # Check current status
        sudo ipmitool sdr type fan
        ```
    *   **Force High Speed (Unofficial/Experimental for iLO):**
        It is often easier to rely on the BIOS "Maximum Cooling" setting. If you must use IPMI to set a static floor:
        (Caution: Improper fan commands can cause overheating)

## 2. BIOS Settings (RBSU)

Reboot the server and press **F9** to enter System Utilities. Configure the following:

1.  **Above 4G Decoding:**
    *   Path: `System Configuration > BIOS/Platform Configuration (RBSU) > PCIe Device Configuration > Above 4G Decoding`.
    *   Setting: **Enabled**. (Essential for GPUs with large VRAM like the 24GB P40).
2.  **Resizable BAR:**
    *   Path: `System Configuration > BIOS/Platform Configuration (RBSU) > PCIe Device Configuration > PCIe Resizable BAR Support`.
    *   Setting: **Enabled** (if available on your specific firmware/CPU combination).
3.  **SR-IOV:**
    *   Path: `System Configuration > BIOS/Platform Configuration (RBSU) > Virtualization Options > SR-IOV`.
    *   Setting: **Enabled**.
4.  **IOMMU (Intel VT-d / AMD IOMMU):**
    *   Ensure Virtualization Technology is **Enabled**.

## 3. Drivers (Ubuntu 22.04 / 24.04)

### Install Dependencies
```bash
sudo apt update
sudo apt install build-essential libglvnd-dev pkg-config
```

### Disable Nouveau (Open Source Driver)
The default `nouveau` driver often conflicts with Tesla cards.

1.  Create a blacklist file:
    ```bash
    sudo bash -c "echo 'blacklist nouveau' > /etc/modprobe.d/blacklist-nouveau.conf"
    sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf"
    ```
2.  Regenerate the kernel initramfs:
    ```bash
    sudo update-initramfs -u
    ```
3.  **Reboot** the server to unload nouveau:
    ```bash
    sudo reboot
    ```

### Install NVIDIA Drivers
Use the proprietary server drivers. The `535-server` or `550-server` branches are stable choices for Compute/AI.

```bash
# Add the graphics drivers PPA (optional, but recommended for latest versions)
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install the driver (choose one)
sudo apt install nvidia-driver-535-server
# OR
sudo apt install nvidia-driver-550-server
```

**Reboot** once more after installation.

## 4. Verification

1.  Check if the GPU is recognized and communicating:
    ```bash
    nvidia-smi
    ```
    *   You should see "Tesla P40" listed.
    *   Check the memory usage and P-state.

2.  **Verify Cooling Performance:**
    Run a load (or use `nvidia-smi` to monitor) and watch the temperature.
    ```bash
    watch -n 1 nvidia-smi
    ```
    *   Idle temp should be 30C-50C.
    *   Load temp should stay under 85C. If it hits 90C+, increase server fan speed immediately.

## Troubleshooting
*   **"Unable to determine the device handle for GPU...":** often means `Above 4G Decoding` is disabled in BIOS.
*   **Card not found (lspci shows nothing):** Check power cabling (EPS vs PCIe issue) or riser seating.
*   **System halts at boot:** Power cabling mismatch (Short circuit protection).
