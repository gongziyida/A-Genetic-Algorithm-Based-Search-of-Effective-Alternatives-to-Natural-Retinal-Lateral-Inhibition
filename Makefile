retina: retina.c retina.h
    gcc -m64 -I${MKLROOT}/include mkl_lab_solution.c -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

clean:
    rm -f retina

# See more information in
# https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/