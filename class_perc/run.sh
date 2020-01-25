#! /bin/bash

if [ ! -f PARAM ] ; then
	echo "Configuration file 'PARAM' does not exist."
	exit 0
fi

# Run the following if there is a linking error
source /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/bin/mklvars.sh intel64

make

echo "Generating data"
./data_generator

if [ ! -d results ] ; then
	mkdir results
fi

# Copy the environment variables and data
for i in PARAM DATA LABELS ;
do
	cp "$i" results/
done

./GA
