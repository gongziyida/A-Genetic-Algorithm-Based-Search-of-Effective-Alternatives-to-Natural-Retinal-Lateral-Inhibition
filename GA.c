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
    double w[MAX_CELLS / 5 + 1]; // Perceptron connection matrix
    double o, coef, err;
    int i, j;

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        // Train
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, MAX_CELLS / 5, w, -1, 1); // Randomize w
        for (j = 0; j < TRAIN_SIZE; j++){ // For each training data
            process(&rps[i], &TRAIN[j * MAX_CELLS]);

            // net = w^T x
            o = cblas_ddot(MAX_CELLS / 5, w, 1,
                    &rps[i].new_states[MAX_CELLS * (rps[i].n_types - 1)], 1);

            o = 1 / (1 + exp(o + w[MAX_CELLS / 5])); // out = sigmoid(net + w_b * bias)

            if (isinf(o)) break;

            if (LABELS_TR[j] == 1) // w -= - eta * 1 / o * o * (1 - o) * input
                coef = - ETA * (1 - o);
            else // w -= - eta * 1 / (1 - o) * o * (1 - o) * input
                coef = - ETA * o;

            cblas_daxpy(MAX_CELLS / 5, coef, &TRAIN[j * MAX_CELLS / 5], 1, w, 1);
            w[MAX_CELLS / 5] -= coef; // Note that the bias is 1 so we do not have to multiply
        }
        // Test
        err = 0;
        for (j = 0; j < TEST_SIZE; j++){ // For each training data
            process(&rps[i], &TEST[j*MAX_CELLS]);

            // net = w^T x
            o = cblas_ddot(MAX_CELLS / 5, w, 1, &rps[i].new_states[MAX_CELLS * (rps[i].n_types - 1)], 1);
            o = 1 / (1 + exp(o + w[MAX_CELLS / 5])); // out = sigmoid(net + w_b * bias)

            if (isinf(o)) break;

            if (LABELS_TR[j] == 1)
                err += -log(o);
            else
                err += -log(1 - o);
        }

        if (isinf(err))
            rps[i].cost = 500;
        else{
            err /= TEST_SIZE;
            rps[i].cost = err + rps[i].total_n_cells / (MAX_TYPES * MAX_CELLS);
        }
    }
}

int comparator(const void *rp1, const void *rp2){
    /*
     * If rp1's cost > rp2's cost (rp1 is worse), rp1 ranks lower than rp2. Return 1.
     * If rp1's cost < rp2's cost (rp2 is worse), rp1 ranks higher than rp2. Return -1.
     * Else, return 0.
     */
    const RetinaParam *rpA = (RetinaParam *) rp1;
    const RetinaParam *rpB = (RetinaParam *) rp2;

    if (rpA->cost > rpB->cost) return 1;
    else if (rpA->cost < rpB->cost) return -1;
    else return 0;
}

void selection(){
    int rival1, rival2;

    for (int i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){ // Choose each child's parents
        // Randomly select two rivals
        rival1 = rand() % NUM_INDIVIDUALS;
        rival2 = rand() % NUM_INDIVIDUALS;

        if (rand() % 100 < 75){ // 0.75 chance to pick the one with the lower cost (winner)
            p1[i] = (rps[rival1].cost < rps[rival2].cost)? rival1 : rival2;
        } else { // 0.25 chance to pick the one with the high cost (loser)
            p1[i] = (rps[rival1].cost < rps[rival2].cost)? rival2 : rival1;
        }

        do{ // Avoid self-crossover
            rival1 = rand() % NUM_INDIVIDUALS;
            rival2 = rand() % NUM_INDIVIDUALS;

        } while (rival1 == p1[i] || rival2 == p1[i]);

        if (rand() % 100 < 75){ // Similar to above
            p2[i] = (rps[rival1].cost < rps[rival2].cost)? rival1 : rival2;
        } else {
            p2[i] = (rps[rival1].cost < rps[rival2].cost)? rival2 : rival1;
        }
    }
}

void crossover(){
    RetinaParam *children = &rps[NUM_ELITES]; // Index start from NUM_ELITES

    int p, n;

    int k;
    for (int i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        if (rand() % 100 < 50)  p = p1[i];
        else                    p = p2[i];

        n = rps[p].n_types;
        children[i].decay = rps[p].decay;
        children[i].n_types = n;

        for (k = 0; k < n; k++){
            if (rand() % 100 < 50)  p = p1[i];
            else                    p = p2[i];

            children[i].axons[k] = rps[p].axons[k];
            children[i].dendrites[k] = rps[p].dendrites[k];
            children[i].polarities[k] = rps[p].polarities[k];
            children[i].n_cells[k] = rps[p].n_cells[k];
        }
    }
}


void mutation(){
    int n, j, chance;
    for (int i = NUM_ELITES; i < NUM_INDIVIDUALS; i++){
        n = rps[i].n_types;

        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                1, &rps[i].decay, rps[i].decay, WIDTH/20.0);
        if (rps[i].decay < 0) rps[i].decay = 0; // Cannot be smaller than 0
        if (rps[i].decay > WIDTH) rps[i].decay = WIDTH; // Cannot be larger than WIDTH

        // Randomly change one of the axon descriptors (same for dendrites)
        rps[i].axons[rand() % n] ^= (int)(rand() - RAND_MAX);
        rps[i].dendrites[rand() % n] ^= (int)(rand() - RAND_MAX);

        // TODO: variable receptors (requires the input to be a function rather than discrete)
        for (j = 1; j < n - 1; j++){ // Skip receptors
            // Mutate polarities, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].polarities[j], rps[i].polarities[j], 0.01);
            // Clip
            if (rps[i].polarities[j] > 1) rps[i].polarities[j] = 1;
            else if (rps[i].polarities[j] < -1) rps[i].polarities[j] = -1;

            chance = rand() % 100;
            // 0.5 probability the number of cells will decrease/increase by 1
            if (chance < 25) {
                rps[i].n_cells[j] += 1;
                if (rps[i].n_cells[j] > MAX_CELLS) rps[i].n_cells[j]--; // Cannot go above the max
            } else if (chance < 50){
                rps[i].n_cells[j] -= 1;
                if (rps[i].n_cells[j] < 0) rps[i].n_cells[j] = 0; // Cannot go below the min (0)
            }
        }
    }
}


int main(int argc, char **argv){
    fprintf(stderr, "Loading data\n");
    load();

    vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    int i, j;
    double max_cost, min_cost, total_cost;

    fprintf(stderr, "Initializing retinas\n");
    // Initialize individual space
    rps = malloc(NUM_INDIVIDUALS * sizeof(RetinaParam));
    for (i = 0; i < NUM_INDIVIDUALS; i++){
        init_retina(&rps[i]);
    }

    // Initialize parent index spaces
    p1 = mkl_malloc(NUM_INDIVIDUALS * sizeof(int), 64);
    p2 = mkl_malloc(NUM_INDIVIDUALS * sizeof(int), 64);

    // Open a log
    FILE *log = fopen("results/LOG", "w");
    fprintf(log, "# MIN MAX AVG\n");

    fprintf(stderr, "Simulation in progress\n");
    for (i = 0; i < MAX_ITERATIONS; i++){
        fprintf(stderr, "\r%d", i);
        fflush(stderr);
        test();

        // Sort the retinas
        qsort(rps, NUM_INDIVIDUALS, sizeof(RetinaParam), comparator);

        // Output stats
        total_cost = 0;
        max_cost = -1;
        min_cost = 999999;
        for (j = 0; j < NUM_INDIVIDUALS; j++){
            total_cost += rps[j].cost;
            if (rps[j].cost > max_cost) max_cost = rps[j].cost;
            if (rps[j].cost < min_cost) min_cost = rps[j].cost;
        }
        fprintf(log, "%d %f %f %f\n", i, min_cost, max_cost, total_cost / NUM_INDIVIDUALS);

        selection();
        crossover();
	    mutation();
	    for (j = 0; j < NUM_INDIVIDUALS; j++){
            mk_connection(&rps[j]);
        }
	}

    // Test and sort the retinas
    test();
    qsort(rps, NUM_INDIVIDUALS, sizeof(RetinaParam), comparator);

	save(rps);

    fprintf(stderr, "\nDone. Removing trashes\n");

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        rm_retina(&rps[i]);
    }

	free_data();

	free(rps);
	mkl_free(p1);
	mkl_free(p2);

	fclose(log);

	return 0;
}