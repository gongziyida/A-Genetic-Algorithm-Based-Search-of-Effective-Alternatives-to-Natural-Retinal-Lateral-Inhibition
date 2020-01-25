#! /bin/bash

if [ ! -f PARAM ] ; then
	echo "Configuration file 'PARAM' does not exist."
	exit 0
fi

# Run the following if there is a linking error
source /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/bin/mklvars.sh intel64

make

if [ ! -d results ] ; then
	mkdir results
fi

# Copy the environment variables and data
cp PARAM results/

./GA
