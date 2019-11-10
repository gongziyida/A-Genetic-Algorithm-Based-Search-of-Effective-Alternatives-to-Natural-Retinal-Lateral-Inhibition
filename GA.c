// Author   Ziyi Gong

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "mkl.h"
#include "retina.h"
#include "io.h"

/* Globals */
int *p1, *p2;       // Parent index spaces
RetinaParam *rps;   // Individual space
VSLStreamStatePtr STREAM; // Random generator stream

void test(){
    double w[MAX_CELLS+1];
    double o, coef, err, d_err;
    int i, j;

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        // Randomize w
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, MAX_CELLS, w, -1, 1);

        for (j = 0; j < TRAIN_SIZE; j++){ // For each training data
            process(&rps[i], &TRAIN[j*MAX_CELLS]);

            cblas_ddot(MAX_CELLS, w, 1, rps[i].new_states, 1); // net = w^T x
            o = tanh(o + w[MAX_CELLS]); // out = tanh(net + w_b * bias)

            d_err = o - LABELS_TR[j];

            // w -= eta * d_err * (1 - o^2) * input
            coef = ETA * d_err * (o * o - 1);
            cblas_daxpy(MAX_CELLS, coef, &TRAIN[j*MAX_CELLS], 1, w, 1);
            w[MAX_CELLS] += coef; // Note that the bias is 1 so we do not have to multiply
        }

        err = 0;

        for (j = 0; j < TEST_SIZE; j++){ // For each training data
            process(&rps[i], &TEST[j*MAX_CELLS]);

            cblas_ddot(MAX_CELLS, w, 1, rps[i].new_states, 1); // net = w^T x
            o = tanh(o + w[MAX_CELLS]); // out = tanh(net + w_b * bias)

            err += (o - LABELS_TR[j]) * (o - LABELS_TR[j]);
        }

        rps[i].score = err;
    }
}

int comparator(const void *rp1, const void *rp2){
    /*
     * If rp1's score > rp2's score (rp1 is worse), rp1 ranks lower rp2. Return value > 0.
     * If rp1's score < rp2's score (rp2 is worse), rp1 ranks higher rp2. Return value < 0.
     * Else, return 0.
     */
    return ((RetinaParam *)rp1)->score - ((RetinaParam *)rp2)->score;
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
    int k;
    for (int i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        if (rand() % 100 < 50) {
            p1i = p1[i];
            p2i = p2[i];
        } else{
            p1i = p2[i];
            p2i = p1[i];
        }

        n = rps[p2i].n_types;

        children[i].decay = rps[p1i].decay;
        children[i].n_types = n;

        for (k = 0; k < MAX_TYPES; k++){
            children[i].axons[k] = rps[p2i].axons[k];
            children[i].dendrites[k] = rps[p2i].dendrites[k];
            children[i].polarities[k] = rps[p2i].polarities[k];
            children[i].n_cells[k] = rps[p2i].n_cells[k];
        }
    }
}


void mutation(){
    int n, j, chance;
    for (int i = NUM_ELITES; i < NUM_INDIVIDUALS; i++){
        n = rps[i].n_types;
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                1, &rps[i].decay, rps[i].decay, WIDTH/20.0);

        // Randomly change one of the axon descriptors (same for dendrites)
        rps[i].axons[rand() % n] ^= (int)(rand() - RAND_MAX);
        rps[i].dendrites[rand() % n] ^= (int)(rand() - RAND_MAX);

        for (j = 1; j < n; j++){ // Skip receptors
            // Mutate polarities, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].polarities[j], rps[i].polarities[j], 0.01);

            chance = rand() % 100;
            // 50% probability the number of cells will decrease/increase by 1
            if (chance < 25) {
                rps[i].n_cells[j] += 1;
                if (rps[i].n_cells[j] > MAX_CELLS) rps[i].n_cells[j]--;
            } else if (chance < 50){
                rps[i].n_cells[j] -= 1;
                if (rps[i].n_cells[j] < 0) rps[i].n_cells[j] = 0;
            }
        }
    }
}


int main(int argc, char **argv){
    printf("Loading data\n");
    load();

    vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    int i, j;

    printf("Initializing retinas\n");
    // Initialize individual space
    rps = malloc(NUM_INDIVIDUALS * sizeof(RetinaParam));
    for (i = 0; i < NUM_INDIVIDUALS; i++){
        init_retina(&rps[i]);
    }

    // Initialize parent index spaces
    p1 = mkl_malloc(NUM_INDIVIDUALS * sizeof(int), 64);
    p2 = mkl_malloc(NUM_INDIVIDUALS * sizeof(int), 64);


    printf("Starting simulations\n");
    for (i = 0; i < MAX_ITERATIONS; i++){
        test();
        selection();
	    crossover();
	    mutation();
	    for (j = 0; j < NUM_INDIVIDUALS; j++){
            mk_connection(&rps[j]);
        }
	}

	// TODO: Output data

	for (i = 0; i < NUM_INDIVIDUALS; i++){
        rm_retina(&rps[i]);
    }

	free_data();

	free(rps);
	mkl_free(p1);
	mkl_free(p2);

	return 0;
}