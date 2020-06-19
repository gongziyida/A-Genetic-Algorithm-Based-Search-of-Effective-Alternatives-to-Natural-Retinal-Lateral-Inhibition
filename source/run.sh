#! /usr/bin/env bash

. ~/intel/compilers_and_libraries_2020.1.217/linux/mkl/bin/mklvars.sh intel64

make release

mkdir -p reg
cp param_regression reg/param
mkdir -p cla
cp param_classification cla/param

echo "Running regression"
./Simulation reg param_regression
python3 visualization.py reg/ 1 0

echo "Running classification"
./Simulation cla param_classification
python3 visualization.py cla/ 1 0
