#!/bin/bash

echo "Running RBF Kernel Simulation..."

echo "Compiling and parsing with increasing feature sizes..."

rm -rf results/rbf_kernel_features_*.txt results/rbf_kernel_datapoints_*.txt


for features in 32 64 128 256; do
    echo "Running with N_FEATURES=$features..."
    make F=$features N=256 
    make F=$features N=256 parse >> "results/rbf_kernel_features_${features}.txt"
done

echo "Compiling and parsing with increasing data points..."
for datapoints in 128 256 512 1024; do
    echo "Running with N_DATAPOINTS=$datapoints..."
    make F=32 N=$datapoints
    make F=32 N=$datapoints parse >> "results/rbf_kernel_datapoints_${datapoints}.txt"
done
