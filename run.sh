#! /bin/sh

if [ ! -f PARAM ] ; then
	echo "Configuration file 'PARAM' does not exist."
	exit 0
fi

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
