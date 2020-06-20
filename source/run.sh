#! /usr/bin/env bash

. ~/intel/compilers_and_libraries_2020.1.217/linux/mkl/bin/mklvars.sh intel64

make release

DIRNAME=noise_lvl_nointernal
mkdir -p ../$DIRNAME

for p in $DIRNAME/*;
do
  folder=../$DIRNAME/${p##*/}
  mkdir -p $folder
  cp $p $folder/param
  ./Simulation $folder $folder/param
  ./visualization.py $folder/ 2 0
done
