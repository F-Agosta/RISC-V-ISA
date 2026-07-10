# RISC-V Custom Stateful Functional Unit for Hyperdimensional Computing

This repository contains the architecture, simulator configurations, and software binaries for evaluating a custom **Stateful Functional Unit (FU)** designed to accelerate Ternary Hyperdimensional Computing (QuantHD) on edge devices. 

The project has transitioned away from traditional floating-point RBF Kernel SVMs, moving entirely to heavily binarized/ternary algorithms to bypass the memory wall and drastically increase instruction per cycle (IPC) efficiency.

## Repository Structure

* **`model/`**: Contains the Python training scripts (`HCD_training.py`). This script trains the QuantHD model and quantizes the hypervectors into Ternary format.
* **`dataset/`**: Contains the raw `.csv` datasets and the generated C-headers (e.g., `HDC_model_har.h`) which pack the trained Ternary Class Prototypes and Bipolar Item Memories for the C kernels.
* **`software/`**: Contains the highly-optimized C baselines and the custom RISC-V software (`kernel_HDC.c` and `kernel_HDC_stateless.c`). The custom software relies on heavily optimized SWAR (SIMD Within A Register) logic.
* **`simulator/`**: Contains the Python configurations for the gem5 `SimpleBoard`. 
  * `hdc_board_stateless.py` configures a standard `MinorCPU` (for the baseline).
  * `hdc_board.py` injects the custom `HdcAlu` Functional Unit into the CPU pipeline to simulate the hardware extensions.
* **`scripts/`**: Contains the master bash script to run the final accurate benchmarks across all datasets, as well as the Python parser used to extract instruction mixes from gem5's raw `stats.txt`.

## How to Run

### 0. Apply the gem5 Patch
Because this project extends the RISC-V ISA with custom HDC instructions (`custom-2` opcode), you must patch a vanilla gem5 installation before running any simulations.
1. Download or clone gem5 (tested on branch `stable`).
2. Apply the patch from the `patches/` directory:
```bash
cd gem5
git apply /path/to/RISC-V-ISA/patches/gem5_hdc_alu.patch
scons build/RISCV/gem5.opt -j$(nproc)
```

### 1. Train the Models (Optional)
If you wish to retrain the QuantHD hypervectors, you can execute the Python training script. This will output new `.h` headers into the `dataset/` directory.
```bash
python3 model/HCD_training.py
```

### 2. Run the Accurate Benchmarks
The repository contains a master benchmark script that automatically loops through the datasets (CANCER, DIGITS, HAR) and simulates them on the cycle-accurate `MinorCPU` edge device. It runs both the Software Baseline and the Stateful Hardware FU.
```bash
./scripts/run_accurate_benchmarks.sh
```

### 3. Parse the Simulation Stats
Once the simulations complete, gem5 will dump massive `stats.txt` files into `data/m5out_accurate_benchmarks/`. To easily extract the execution times, IPC, and custom instruction counts (like the `HdcAlu` workload), run the parsing script:
```bash
python3 scripts/parse_stats.py
```
