// Author   Ziyi Gong
// Version  0.1
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define

/* Globals */
int MAX_ITERATIONS, NUM_INDIVIDUALS, SELECTIVITY;

int comparator(const void *rp1, const void *rp2){
    /*
     * If rp1->score > rp2->score, rp1 should go before rp2. Return value < 0.
     * If rp1->score < rp2->score, rp1 should go after rp2. Return value > 0.
     * Else, return 0.
     */
    return rp2->score - rp1->score;
}

void selection(RetinaParam **rps){
    // Sort the retinas
	qsort(*rps, NUM_INDIVIDUALS, sizeof(RetinaParam), comparator);
}

void crossover(RetinaParam **rps){
    // TODO: Implement crossover over all parents, with chance by score or rank
}


void mutation(RetinaParam **rps){
    // TODO: Define the range/attribute of mutation on each parameter and implement the mutation
}

void run(){
    // TODO: Integrate the above
}


int main(int argc, char **argv){
	if (argc > 4){
		fprintf(stderr,
			"retina <MAX ITERATIONS> <NUM INDIVIDUALS> <WIDTH>\n");
		exit(EXIT_FAILURE);
	}

	/* Initialization */
	printf("Initializing...\n");

	/* Constant declarations begin */
	
	// Maximum number of iterations allowed
	if (argc > 1)	MAX_ITERATIONS = atoi(argv[1]);
	else 			MAX_ITERATIONS = 1000;
	
	// Number of individuals in each epoch
	if (argc > 2)	NUM_INDIVIDUALS = atoi(argv[2]);
	else 			NUM_INDIVIDUALS = 100;

	// Width of the retina
	if (argc > 3)	WIDTH = atoi(argv[4]);
	else 			WIDTH = 50;


	return 0;
}