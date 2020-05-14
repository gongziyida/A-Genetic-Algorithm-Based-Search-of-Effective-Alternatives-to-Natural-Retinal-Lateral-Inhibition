CFLAGS	= -g -Wall -Werror

OBJS	= $(patsubst %, obj/%, main.o, Retina.o, GA.o)

DEPS 	= Retina.h GA.h

Simulation: $(OBJS) $(DEPS)
	g++ $(CFLAGS) $(OBJS) -o $@ 

obj/%.o: %.c $(DEPS)
	@ mkdir -p obj # Mkdir if not found
	g++ $(CFLAGS) -c -o $@ $<

clean:
	rm -rf obj Simulation
