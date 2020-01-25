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

/* Test on perturbed stimuli using perceptron */
void test_p_p(){
    double ref[MAX_CELLS], err;
    int i;

    // Define reference signal
    for (i = 0; i < MAX_CELLS / 2; i++) ref[i] = 0;
    for (i = MAX_CELLS / 2; i < MAX_CELLS; i++) ref[i] = 1;

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        process(&rps[i], ref);
        rps[i].cost = 0;

        int j = MAX_CELLS / 2;
        int k = j - 1;
        double left = 0, right = 0;
        for (; j < MAX_CELLS && k >= 0; j++, k--){
            left += rps[i].new_states[k];
            right += rps[i].new_states[j];
        }
        left /= 15;
        right /= 15;
        rps[i].cost = fabs(right - left) * 0.1;


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

int select_p(int cur, int another_p){
    int p, rival1, rival2;
    do {// Randomly select two rivals
        rival1 = rand() % NUM_INDIVIDUALS;
        rival2 = rand() % NUM_INDIVIDUALS;
    } while (rival1 == cur || rival2 == cur || rival1 == another_p || rival2 == another_p);

    double ratio = rps[rival1].cost / rps[rival2].cost;
    if (ratio > 1) ratio = 1.0 / ratio; // Assume the range is (0, 1]

    if (rand() % 100 < 100 - 50 * ratio){ // Pick the one with the lower cost (winner)
        p = (rps[rival1].cost < rps[rival2].cost)? rival1 : rival2;
    } else { // Pick the one with the high cost (loser)
        p = (rps[rival1].cost < rps[rival2].cost)? rival2 : rival1;
    }

    return p;
}

void selection(){
    for (int i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){ // Choose each child's parents
        p1[i] = select_p(i + NUM_ELITES, -1);
        p2[i] = select_p(i + NUM_ELITES, p1[i]);
    }
}

void crossover(){
    RetinaParam children[NUM_INDIVIDUALS - NUM_ELITES]; // buffer

    int p, n, q;
    int i, k;
    for (i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        if (rand() % 100 < 50)  p = p1[i];
        else                    p = p2[i];

        n = rps[p].n_types;

        for (k = 0; k < n; k++){
            if (rand() % 100 < 50)  p = p1[i];
            else                    p = p2[i];

            if (k == 0) // Receptor cells
                q = 0;
            else { // Interneurons
                if (rps[p].n_types > 1) // Has interneurons
                    q = 1 + rand() % (rps[p].n_types - 1);
                else {
                    p = p == p1[i] ? p2[i] : p1[i]; // Another parent must have interneurons
                    q = 1 + rand() % (rps[p].n_types - 1);
                }
            }

            children[i].axons[k] = rps[p].axons[q];
            children[i].dendrites[k] = rps[p].dendrites[q];
            children[i].polarities[k] = rps[p].polarities[q];
            children[i].n_cells[k] = rps[p].n_cells[q];
            children[i].phi[k] = rps[p].phi[q];
            children[i].beta[k] = rps[p].beta[q];
        }

//        children[i].decay = rps[p].decay;
        children[i].n_types = n;
    }

    for (i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        rps[NUM_ELITES + i].n_types = children[i].n_types;

        for (k = 0; k < MAX_TYPES; k++){
            rps[NUM_ELITES + i].axons[k] = children[i].axons[k];
            rps[NUM_ELITES + i].dendrites[k] = children[i].dendrites[k];
            rps[NUM_ELITES + i].polarities[k] = children[i].polarities[k];
            rps[NUM_ELITES + i].n_cells[k] = children[i].n_cells[k];
            rps[NUM_ELITES + i].phi[k] = children[i].phi[k];
            rps[NUM_ELITES + i].beta[k] = children[i].beta[k];
        }
    }
}


void mutation(){
    int n, i, j, chance, k;
    for (i = NUM_ELITES; i < NUM_INDIVIDUALS; i++){
        n = rps[i].n_types;

//        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
//                1, &rps[i].decay, rps[i].decay, WIDTH/20.0);
//        if (rps[i].decay < 0) rps[i].decay = 0; // Cannot be smaller than 0
//        if (rps[i].decay > WIDTH) rps[i].decay = WIDTH; // Cannot be larger than WIDTH

        // Randomly flip two bits for each axon and dendrite descriptor
        for (j = 0; j < n; j++){
            for (k = 0; k < 2; k++){
                rps[i].axons[j] ^= 1 << (rand() % 32);
                rps[i].dendrites[j] ^= 1 << (rand() % 32);
            }
        }

        for (j = 0; j < n; j++){
            // Mutate polarities, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].polarities[j], rps[i].polarities[j], 0.001);
            // Clip
            if (rps[i].polarities[j] > 1) rps[i].polarities[j] = 1;
            else if (rps[i].polarities[j] < -1) rps[i].polarities[j] = -1;
            else if (rps[i].polarities[j] == 0) rps[i].polarities[j] = 0.01; // in case of 0

            // Mutate phi, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].phi[j], rps[i].phi[j], 0.1);
            // Clip
            if (rps[i].phi[j] > WIDTH / 2) rps[i].phi[j] = WIDTH / 2;
            else if (rps[i].phi[j] < 1) rps[i].phi[j] = 1;

            // Mutate beta, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].beta[j], rps[i].beta[j], 0.1);
            // Clip
            if (rps[i].beta[j] > WIDTH / 2) rps[i].beta[j] = WIDTH / 2;
            else if (rps[i].beta[j] < 0) rps[i].beta[j] = 0;

            // Skip receptors
            if (j == 0) continue;

            chance = rand() % 100;
            // 0.05 probability the number of cells will decrease/increase by 1
            if (chance < 5) {
                rps[i].n_cells[j] += 1;
                // Cannot go below the min (0)
                if (rps[i].n_cells[j] > MAX_CELLS) rps[i].n_cells[j] = MAX_CELLS;
            } else if (chance < 10){
                rps[i].n_cells[j] -= 1;
                // Cannot go below the min (0)
                if (rps[i].n_cells[j] < 0) rps[i].n_cells[j] = 0;
            }
        }

        // Force the receptors to have excitatory projections
        if (rps[i].polarities[0] < 0) rps[i].polarities[0] = fabs(rps[i].polarities[0]);
    }
}

//int main(int argc, char **argv){ // Testing main
//    load();
//
//    vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);
//
//    // Initialize individual space
//    rps = malloc(sizeof(RetinaParam));
//    init_retina(rps);
//    process(rps, TRAIN);
//
//    save(rps);
//
//	free_data();
//
//	free(rps);
//
//	return 0;
//}

int main(int argc, char **argv){
    void (*test)() = test_p_p; // For convenience

    fprintf(stderr, "Loading data\n");
    load();

    vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    int i, j;

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

    fprintf(stderr, "Simulation in progress\n");
    for (i = 0; i < MAX_ITERATIONS; i++){
        fprintf(stderr, "\r%d", i);
        fflush(stderr);
        test();

        // Sort the retinas
        qsort(rps, NUM_INDIVIDUALS, sizeof(RetinaParam), comparator);

        // Output stats
        for (j = 0; j < NUM_INDIVIDUALS; j++) fprintf(log, "%f ", rps[j].cost);
        fprintf(log, "\n");

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

	free(rps);
	mkl_free(p1);
	mkl_free(p2);

	fclose(log);

	return 0;
}