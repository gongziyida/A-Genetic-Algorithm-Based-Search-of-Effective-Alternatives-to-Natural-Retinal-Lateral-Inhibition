#! /usr/bin/env bash

. ~/intel/compilers_and_libraries_2020.1.217/linux/mkl/bin/mklvars.sh intel64

make release

mkdir -p test
cp param_template test/param

./Simulation test param_template
python visualization.py test/ 0 0
