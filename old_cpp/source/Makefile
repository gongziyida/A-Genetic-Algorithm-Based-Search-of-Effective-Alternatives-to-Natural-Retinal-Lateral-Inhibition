CFLAGS	= -std=c++17 -march=native -fopenmp -Wno-unused-result -Wall -Werror -Wextra

OBJS	= tool.o Retina.o GA.o main.o

OBJSD	= $(addprefix .obj/, $(OBJS))

DEPS 	= tool.h Retina.h GA.h

INCLUDES= -I/usr/include/eigen3 -I${MKLROOT}/include -I.

LDFLAGS	= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

LDLIBS	= -lm -ldl

COMPFLAGS = -DMKL_ILP64 -m64 -I${MKLROOT}/include

.obj/%.o: %.cpp $(DEPS)
	@ mkdir -p .obj
	g++ $(CFLAGS) $(COMPFLAGS) $(INCLUDES) -c -o $@ $<

Simulation: $(OBJSD) $(DEPS)
	g++ $(CFLAGS) $(OBJSD) $(LDFLAGS) $(LDLIBS) -o $@ 
	
debug: CFLAGS += -g
debug: Simulation

release: CFLAGS += -O3
release: Simulation

clean:
	rm -rf .obj/ Simulation
