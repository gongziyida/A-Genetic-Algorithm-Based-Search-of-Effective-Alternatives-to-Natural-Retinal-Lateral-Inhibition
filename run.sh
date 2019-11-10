#! /bin/sh

if [ ! -f PARAM ] ; then
	echo "Configuration file 'PARAM' does not exist."
	exit 0
fi

# Run the following if there is a linking error
# source /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/bin/mklvars.sh intel64

if [ ! -f GA ] ; then
	make
fi

if [ ! -f DATA ] || [ ! -f LABELS ] ; then
	echo "Generating data"
	./data_generator
fi

# Output the results to the user-specified file
if [ -z "$1" ] ; then
	echo "Results will be printed to STDOUT"
	./GA
else
	echo "Results will be saved to '$1'"
	./GA > $1
fi
