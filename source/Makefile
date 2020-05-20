CFLAGS	= -std=c++11 -march=native -fopenmp -Wno-unused-result -Wall -Werror -Wextra

OBJS	= tool.o Retina.o GA.o main.o

OBJSD	= $(addprefix .obj/, $(OBJS))

DEPS 	= tool.h Retina.h GA.h


.obj/%.o: %.cpp $(DEPS)
	@ mkdir -p .obj
	g++ $(CFLAGS) -c -o $@ $<

Simulation: $(OBJSD) $(DEPS)
	g++ $(CFLAGS) $(OBJSD) -o $@ 
	
debug: CFLAGS += -g
debug: Simulation

release: CFLAGS += -O3
release: Simulation

clean:
	rm -rf .obj/ Simulation
