source /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/bin/mklvars.sh intel64

make release

./Simulation

python3 visualization.py results/ 2 0
