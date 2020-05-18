CFLAGS	= -g 

OBJS	= tool.o Retina.o GA.o

# OBJSD	= $(addprefix .obj/, $(OBJS))

DEPS 	= tool.h Retina.h GA.h


.obj/%.o: %.cpp $(DEPS)
	@ mkdir -p .obj
	g++ $(CFLAGS) -c -o $@ $<

Simulation: .obj/$(OBJS) $(DEPS)
	g++ $(CFLAGS) .obj/$(OBJS) -o $@ 
	

clean:
	rm -rf .obj/ Simulation
