# On how to compile, see more information in
# https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/

CC	= gcc

CFLAGS	= -g -Wall -Werror -m64

OBJS	= $(patsubst %, obj/%, io.o retina.o GA.o)

INCLUDES= -I${MKLROOT}/include -I.

LDFLAGS	= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

LDLIBS	= -lm -ldl

DEPS	= retina.h io.h

obj/%.o: %.c $(DEPS)
	@ mkdir -p obj
	$(CC) $(CFLAGS) -c -o $@ $<


GA: $(OBJS) $(DEPS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(INCLUDES) $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf GA obj DATA LABELS
