#!/bin/bash

GEM5_PATH="/Users/francescoagosta/ACA_project/gem5"
DATASETS=("CANCER" "DIGITS" "HAR")

mkdir -p data/m5out_accurate_benchmarks

for ds in "${DATASETS[@]}"; do
    echo "=================================================="
    echo "Simulating Baseline (Accurate FU) - Dataset: ${ds}"
    outdir="data/m5out_accurate_benchmarks/baseline_${ds}"
    mkdir -p ${outdir}
    ${GEM5_PATH}/build/RISCV/gem5.opt -d ${outdir} simulator/hdc_board.py "software/build/kernel_HDC_${ds}.elf" minor > /dev/null

    echo "Simulating Stateless (Accurate FU) - Dataset: ${ds}"
    outdir_stateless="data/m5out_accurate_benchmarks/stateless_${ds}"
    mkdir -p ${outdir_stateless}
    ${GEM5_PATH}/build/RISCV/gem5.opt -d ${outdir_stateless} simulator/hdc_board_stateless.py "software/build/kernel_HDC_stateless_${ds}.elf" minor > /dev/null
done

echo "=================================================="
echo "Accurate Benchmarks Finished. Parsing Results..."
echo "=================================================="

for ds in "${DATASETS[@]}"; do
    base_ticks=$(grep "^simTicks " data/m5out_accurate_benchmarks/baseline_${ds}/stats.txt | head -n 1 | awk '{print $2}')
    state_ticks=$(grep "^simTicks " data/m5out_accurate_benchmarks/stateless_${ds}/stats.txt | head -n 1 | awk '{print $2}')
    
    echo "Dataset ${ds}:"
    echo "  Baseline:  ${base_ticks} Ticks"
    echo "  Stateless: ${state_ticks} Ticks"
    echo "--------------------------------------------------"
done
