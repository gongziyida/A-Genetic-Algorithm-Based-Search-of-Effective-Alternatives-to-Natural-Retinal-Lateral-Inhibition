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
#include <string.h>

#define

/* Globals */
int MAX_ITERATIONS, NUM_INDIVIDUALS, NUM_ELITES;
RetinaParam *rps;   // Individual space
int *p1, *p2;       // Parent index spaces
VSLStreamStatePtr STREAM; // Random generator stream

int comparator(const void *rp1, const void *rp2){
    /*
     * If rp1->score > rp2->score, rp1 should go before rp2. Return value < 0.
     * If rp1->score < rp2->score, rp1 should go after rp2. Return value > 0.
     * Else, return 0.
     */
    return rp2->score - rp1->score;
}

void selection(){
    int rival1, rival2;

    for (int i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        rival1 = rand() % NUM_INDIVIDUALS;
        rival2 = rand() % NUM_INDIVIDUALS;

        if (rand() % 100 < 75){
            p1[i] = (rps[rival1].score > rps[rival2].score)? rival1 : rival2;
        } else {
            p1[i] = (rps[rival1].score > rps[rival2].score)? rival2 : rival1;
        }

        do{ // Avoid self-crossover
            rival1 = rand() % NUM_INDIVIDUALS;
            rival2 = rand() % NUM_INDIVIDUALS;

        } while (rival1 == p1[i] || rival2 == p1[i]);

        if (rand() % 100 < 75){
            p2[i] = (rps[rival1].score > rps[rival2].score)? rival1 : rival2;
        } else {
            p2[i] = (rps[rival1].score > rps[rival2].score)? rival2 : rival1;
        }
    }
}

void crossover(){
    // Sort the retinas
    qsort(rps, NUM_INDIVIDUALS, sizeof(RetinaParam), comparator);

    RetinaParam *children = &rps[NUM_ELITES]; // Index start from NUM_ELITES

    int p1i, p2i, n;

    // Single-point crossover
    for (int i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        if (rand() % 100 < 50) {
            p1i = p1[i];
            p2i = p2[i];
        } else{
            p1i = p2[i];
            p2i = p1[i];
        }

        n = rps[p2i].n_types;

        rps[i+NUM_ELITES].decay = rps[p1i].decay;
        rps[i+NUM_ELITES].n_types = n;
        strncpy(rps[i+NUM_ELITES].axons, rps[p2i].axons, MAX_TYPES * N_FACTORS);
        strncpy(rps[i+NUM_ELITES].dendrites, rps[p2i].dendrites, MAX_TYPES * N_FACTORS);
        strncpy(rps[i+NUM_ELITES].n_cells, rps[p2i].n_cells, MAX_TYPES);
    }
}


void mutation(){
    double buff[rps[i+NUM_ELITES].n_types * N_FACTORS];
    for (int i = NUM_ELITES; i < NUM_INDIVIDUALS; i++){
        if (rand() % 100 < 20){ // Mutate decay with probability 0.2
            (VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                    1, rps[i+NUM_ELITES].decay, rps[i+NUM_ELITES].decay, WIDTH/20);
        }

        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                      rps[i+NUM_ELITES].n_types * N_FACTORS, buff, 0, 0.01);

    }
}


int main(int argc, char **argv){
	if (argc > 5){
		fprintf(stderr,
			"retina <MAX ITERATIONS> <NUM INDIVIDUALS> <NUM_ELITES> <WIDTH>\n");
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

	// Number of individuals in each epoch
    if (argc > 3)	NUM_ELITES = atoi(argv[3]);
    else 			NUM_ELITES = 25;

    vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    int i;

	// Make individual space
	RetinaParam *rps = malloc(NUM_INDIVIDUALS * sizeof(RetinaParam));
	for (i = 0; i < NUM_INDIVIDUALS; i++){
	    mk_retina(&rps[i]);
	}

	for (i = 0; i < MAX_ITERATIONS; i++){
	    selection();
	    crossover();
	    mutation();
	}

	for (i = 0; i < NUM_INDIVIDUALS; i++){
        rm_retina(&rps[i]);
    }
	free(rps);

	return 0;
}