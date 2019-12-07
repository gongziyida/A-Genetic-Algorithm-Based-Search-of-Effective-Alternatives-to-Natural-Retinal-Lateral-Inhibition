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
    double w[MAX_CELLS / 5 + 1]; // Perceptron connection matrix
    double o, coef, err, max_cost;
    double max_w, min_w;
    int i, j, k;
    int should_die;

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        max_cost = INT_MIN;
        err = 0;
        should_die = 0;

        // TODO: Solve the over train problem
        // Train
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, MAX_CELLS / 5 + 1, w, -1, 1); // Randomize
        for (j = 0; j < TRAIN_SIZE; j++){ // For each training data
            process(&rps[i], &TRAIN[j * MAX_CELLS]);
//            fprintf(stderr, "\n");
//            for (int l = 0; l < MAX_CELLS / 5; l++)
//            fprintf(stderr, "%f ", rps[i].new_states[MAX_CELLS * (rps[i].n_types - 1) + l]);

            // net = w^T x
            o = cblas_ddot(MAX_CELLS / 5, w, 1,
                    &rps[i].new_states[MAX_CELLS * (rps[i].n_types - 1)], 1);

            /* For testing purpose
            for (int l = 0; l < rps[i].n_types; l++){
                fprintf(stderr, "\n%d\n", l);
                for (int k = 0; k < rps[i].n_cells[l]; k++){
                    if (isnan(rps[i].new_states[l * MAX_CELLS + k]))
                        fprintf(stderr, "%f ", rps[i].new_states[l * MAX_CELLS + k]);
                }
            }
             */

            if (isinf(o)){
                should_die = 1;
                break;
            }

            // Because it is possible for a ill-behaved retina to give zero output which easily
            // leads the perceptron to have near zero output, we cannot use sigmoid directly here
            o = tanh(o);// + w[MAX_CELLS / 5]); // out = tanh(net + w_b * bias)
            o = (o + 1) / 2; // Map (-1, 1) to (0, 1), so as to avoid division by zero

            if (LABELS_TR[j] == 1) { // w -= - eta * (1 - o^2) / o * input
                if (o < 0.001) o = 0.001; // Avoid near 0
                coef = -ETA * (1 - o * o) / o;
            } else { // w -= - eta * (1 - o^2) / (1 - o) * input
                if (o > 0.999) o = 0.999; // Avoid near 1
                coef = -ETA * (1 - o * o) / (1 - o);
            }

            cblas_daxpy(MAX_CELLS / 5, coef,
                    &rps[i].new_states[MAX_CELLS * (rps[i].n_types - 1)], 1, w, 1);
//            w[MAX_CELLS / 5] -= coef; // Note that the bias is 1 so we do not have to multiply

            // Find min/max
            for (k = 0, min_w = 1e+308, max_w = -1e+308; k < MAX_CELLS / 5; k++){
                if (w[k] < min_w) min_w = w[k];
                if (w[k] > max_w) max_w = w[k];
            }
            // Normalize
            for (k = 0; k < MAX_CELLS / 5; k++) w[k] = (w[k] - min_w) / (max_w - min_w);
        }

        // Test
        for (j = 0; j < TEST_SIZE; j++){ // For each training data
            if (should_die) break;
            process(&rps[i], &TEST[j*MAX_CELLS]);

            // net = w^T x
            o = cblas_ddot(MAX_CELLS / 5, w, 1,
                    &rps[i].new_states[MAX_CELLS * (rps[i].n_types - 1)], 1);
            o = tanh(o);// + w[MAX_CELLS / 5]); // out = tanh(net + w_b * bias)
            o = (o + 1) / 2; // Map (-1, 1) to (0, 1), so as to avoid division by zero

            if (LABELS_TR[j] == 1)
                err += -log(o < 0.001 ? 0.001 : o);
            else
                err += -log(o > 0.999 ? 0.001 : (1 - o));
        }

        if (should_die || isinf(err)) {
            rps[i].cost = INT_MAX;
            should_die = 0;
        } else{
            err /= TEST_SIZE;
            rps[i].cost = err;
//            rps[i].cost = err + 1 / rps[i].avg_intvl;
//            rps[i].cost = err + 0.05 * rps[i].n_layers;
//            rps[i].cost = err + 1 / rps[i].avg_intvl + 0.05 * rps[i].n_layers;
//            rps[i].cost = err + (double) rps[i].n_synapses / (2 * pow(MAX_CELLS * MAX_TYPES, 2));
            if (rps[i].cost > max_cost) max_cost = rps[i].cost;
        }
    }

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        if (rps[i].cost == INT_MAX) rps[i].cost = max_cost + 1;
    }
}

// TODO: Try the following:
void test_bam_p(); // perturbed stimuli
void test_fft_p(); // blurred stimuli

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
    RetinaParam *children = &rps[NUM_ELITES]; // Index start from NUM_ELITES

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
            else if (k == n - 1) // Ganglion cells
                q = rps[p].n_types - 1;
            else { // Interneurons
                if (rps[p].n_types > 2) // Has interneurons
                    q = 1 + rand() % (rps[p].n_types - 2);
                else {
                    p = p == p1[i] ? p2[i] : p1[i]; // Another parent must have interneurons
                    q = 1 + rand() % (rps[p].n_types - 2);
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

        // TODO: variable receptors (requires the input to be a function rather than discrete)
        for (j = 0; j < n; j++){
            // Mutate polarities, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].polarities[j], rps[i].polarities[j], 0.001);
            // Clip
            if (rps[i].polarities[j] > 1) rps[i].polarities[j] = 1;
            else if (rps[i].polarities[j] < -1) rps[i].polarities[j] = -1;

            // Mutate phi, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].phi[j], rps[i].phi[j], 0.1);
            // Clip
            if (rps[i].phi[j] > WIDTH) rps[i].phi[j] = WIDTH;
            else if (rps[i].phi[j] < 1) rps[i].phi[j] = 1;

            // Mutate beta, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rps[i].beta[j], rps[i].beta[j], 0.1);
            // Clip
            if (rps[i].beta[j] > WIDTH) rps[i].beta[j] = WIDTH;
            else if (rps[i].beta[j] < 0) rps[i].beta[j] = 0;

            // Skip receptors and ganglion cells
            if (j == 0 || j == n - 1) continue;

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

	free_data();

	free(rps);
	mkl_free(p1);
	mkl_free(p2);

	fclose(log);

	return 0;
}