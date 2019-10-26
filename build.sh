#! /bin/sh
if [ -z "$MKLROOT" ]
then
	echo "Set up environment variables\n"
	source /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/bin/mklvars.sh intel64
fi

make
