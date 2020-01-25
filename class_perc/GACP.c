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
Retina *rs;   // Individual space
Retina *children;   // buffer space

VSLStreamStatePtr STREAM; // Random generator stream

/* Test on perturbed stimuli using perceptron */
void test(){
    double w[NUM_RGCS + 1]; // Perceptron connection matrix
    double o, coef, err, max_cost;
    double max_w, min_w;
	double gang_out[NUM_RGCS];
    int i, j, k;
    int should_die;

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        max_cost = INT_MIN;
        err = 0;
        should_die = 0;

        // TODO: Solve the over train problem
        // Train
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, NUM_RGCS + 1, w, -1, 1); // Randomize
        for (j = 0; j < TRAIN_SIZE; j++){ // For each training data
            process(&rs[i], &TRAIN[j * MAX_CELLS], gang_out);

            // net = w^T x
            o = cblas_ddot(NUM_RGCS, w, 1, gang_out, 1);

            if (isinf(o)){
                should_die = 1;
                break;
            }

            // Because it is possible for a ill-behaved retina to give zero output which easily
            // leads the perceptron to have near zero output, we cannot use sigmoid directly here
            o = tanh(o);// + w[NUM_RGCS]); // out = tanh(net + w_b * bias)
            o = (o + 1) / 2; // Map (-1, 1) to (0, 1), so as to avoid division by zero

            if (LABELS_TR[j] == 1) { // w -= - eta * (1 - o^2) / o * input
                if (o < 0.001) o = 0.001; // Avoid near 0
                coef = -ETA * (1 - o * o) / o;
            } else { // w -= - eta * (1 - o^2) / (1 - o) * input
                if (o > 0.999) o = 0.999; // Avoid near 1
                coef = -ETA * (1 - o * o) / (1 - o);
            }

            cblas_daxpy(NUM_RGCS, coef, gang_out, 1, w, 1);
//            w[NUM_RGCS] -= coef; // Note that the bias is 1 so we do not have to multiply

            // Find min/max
            for (k = 0, min_w = 1e+308, max_w = -1e+308; k < NUM_RGCS; k++){
                if (w[k] < min_w) min_w = w[k];
                if (w[k] > max_w) max_w = w[k];
            }
            // Normalize
            for (k = 0; k < NUM_RGCS; k++) w[k] = (w[k] - min_w) / (max_w - min_w);
        }

        // Test
        for (j = 0; j < TEST_SIZE; j++){ // For each training data
            if (should_die) break;
            process(&rs[i], &TEST[j*MAX_CELLS], gang_out);

            // net = w^T x
            o = cblas_ddot(NUM_RGCS, w, 1, gang_out, 1);
            o = tanh(o);// + w[NUM_RGCS]); // out = tanh(net + w_b * bias)
            o = (o + 1) / 2; // Map (-1, 1) to (0, 1), so as to avoid division by zero

            if (LABELS_TR[j] == 1)
                err += -log(o < 0.001 ? 0.001 : o);
            else
                err += -log(o > 0.999 ? 0.001 : (1 - o));
        }

        if (should_die || isinf(err)) {
            rs[i].cost = INT_MAX;
            should_die = 0;
        } else{
            err /= TEST_SIZE;
            rs[i].cost = err;
//            rs[i].cost = err + 1 / rs[i].avg_intvl;
//            rs[i].cost = err + (double) rs[i].n_synapses / (2 * pow(MAX_CELLS * MAX_TYPES, 2));
            if (rs[i].cost > max_cost) max_cost = rs[i].cost;
        }
    }

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        if (rs[i].cost == INT_MAX) rs[i].cost = max_cost + 1;
    }
}


int comparator(const void *r1, const void *r2){
    /*
     * If r1's cost > r2's cost (r1 is worse), r1 ranks lower than r2. Return 1.
     * If r1's cost < r2's cost (r2 is worse), r1 ranks higher than r2. Return -1.
     * Else, return 0.
     */
    const Retina *rA = (Retina *) r1;
    const Retina *rB = (Retina *) r2;

    if (rA->cost > rB->cost) return 1;
    else if (rA->cost < rB->cost) return -1;
    else return 0;
}

int select_p(int cur, int another_p){
    int p, rival1, rival2;
    do {// Randomly select two rivals
        rival1 = rand() % NUM_INDIVIDUALS;
        rival2 = rand() % NUM_INDIVIDUALS;
    } while (rival1 == cur || rival2 == cur || rival1 == another_p || rival2 == another_p);

    double ratio = rs[rival1].cost / rs[rival2].cost;
    if (ratio > 1) ratio = 1.0 / ratio; // Assume the range is (0, 1]

    if (rand() % 100 < 100 - 50 * ratio){ // Pick the one with the lower cost (winner)
        p = (rs[rival1].cost < rs[rival2].cost)? rival1 : rival2;
    } else { // Pick the one with the high cost (loser)
        p = (rs[rival1].cost < rs[rival2].cost)? rival2 : rival1;
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
    int p, n, q;
    int i, k;
    for (i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        if (rand() % 100 < 50)  p = p1[i];
        else                    p = p2[i];

        n = rs[p].n_types;

        for (k = 0; k < n; k++){
            if (rand() % 100 < 50)  p = p1[i];
            else                    p = p2[i];

            if (k == 0) // Receptor cells
                q = 0;
            else if (k == n - 1) // Ganglion cells
                q = rs[p].n_types - 1;
            else { // Interneurons
                if (rs[p].n_types > 2) // Has interneurons
                    q = 1 + rand() % (rs[p].n_types - 2);
                else {
                    p = p == p1[i] ? p2[i] : p1[i]; // Another parent must have interneurons
                    q = 1 + rand() % (rs[p].n_types - 2);
                }
            }

            children[i].axons[k] = rs[p].axons[q];
            children[i].dendrites[k] = rs[p].dendrites[q];
            children[i].polarities[k] = rs[p].polarities[q];
            children[i].n_cells[k] = rs[p].n_cells[q];
            children[i].phi[k] = rs[p].phi[q];
            children[i].beta[k] = rs[p].beta[q];
        }

//        children[i].decay = rs[p].decay;
        children[i].n_types = n;
    }

    for (i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        rs[NUM_ELITES + i].n_types = children[i].n_types;

        for (k = 0; k < MAX_TYPES; k++){
            rs[NUM_ELITES + i].axons[k] = children[i].axons[k];
            rs[NUM_ELITES + i].dendrites[k] = children[i].dendrites[k];
            rs[NUM_ELITES + i].polarities[k] = children[i].polarities[k];
            rs[NUM_ELITES + i].n_cells[k] = children[i].n_cells[k];
            rs[NUM_ELITES + i].phi[k] = children[i].phi[k];
            rs[NUM_ELITES + i].beta[k] = children[i].beta[k];
        }
    }
}


void mutation(){
    int n, i, j, chance, k;
    for (i = NUM_ELITES; i < NUM_INDIVIDUALS; i++){
        n = rs[i].n_types;

//        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
//                1, &rs[i].decay, rs[i].decay, WIDTH/20.0);
//        if (rs[i].decay < 0) rs[i].decay = 0; // Cannot be smaller than 0
//        if (rs[i].decay > WIDTH) rs[i].decay = WIDTH; // Cannot be larger than WIDTH

        // Randomly flip two bits for each axon and dendrite descriptor
        for (j = 0; j < n; j++){
            for (k = 0; k < 2; k++){
                rs[i].axons[j] ^= 1 << (rand() % 32);
                rs[i].dendrites[j] ^= 1 << (rand() % 32);
            }
        }

        for (j = 0; j < n; j++){
            // Mutate polarities, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rs[i].polarities[j], rs[i].polarities[j], 0.001);
            // Clip
            if (rs[i].polarities[j] > 1) rs[i].polarities[j] = 1;
            else if (rs[i].polarities[j] < -1) rs[i].polarities[j] = -1;
            else if (rs[i].polarities[j] == 0) rs[i].polarities[j] = 0.01; // in case of 0

            // Mutate phi, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rs[i].phi[j], rs[i].phi[j], 0.1);
            // Clip
            if (rs[i].phi[j] > WIDTH / 2) rs[i].phi[j] = WIDTH / 2;
            else if (rs[i].phi[j] < 1) rs[i].phi[j] = 1;

            // Mutate beta, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rs[i].beta[j], rs[i].beta[j], 0.1);
            // Clip
            if (rs[i].beta[j] > WIDTH / 2) rs[i].beta[j] = WIDTH / 2;
            else if (rs[i].beta[j] < 0) rs[i].beta[j] = 0;

            // Skip receptors and ganglion cells
            if (j == 0 || j == n - 1) continue;

            chance = rand() % 100;
            // 0.05 probability the number of cells will decrease/increase by 1
            if (chance < 5) {
                rs[i].n_cells[j] += 1;
                // Cannot go below the min (0)
                if (rs[i].n_cells[j] > MAX_CELLS) rs[i].n_cells[j] = MAX_CELLS;
            } else if (chance < 10){
                rs[i].n_cells[j] -= 1;
                // Cannot go below the min (0)
                if (rs[i].n_cells[j] < 0) rs[i].n_cells[j] = 0;
            }
        }

        // Force the receptors to have excitatory projections
        if (rs[i].polarities[0] < 0) rs[i].polarities[0] = fabs(rs[i].polarities[0]);
    }
}

int main(int argc, char **argv){
    fprintf(stderr, "Loading data\n");
    load();

    vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    int i, j;

    fprintf(stderr, "Initializing retinas\n");

    // Initialize individual space
	Retina aux_rs[NUM_INDIVIDUALS];
	rs = aux_rs;

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        maker(&rs[i]);
    }

    // Initialize parent index spaces
	int aux_p1[NUM_INDIVIDUALS], aux_p2[NUM_INDIVIDUALS];
	p1 = aux_p1;
    p2 = aux_p2;

	// Initialize children
	Retina aux_c[NUM_INDIVIDUALS - NUM_ELITES]; 
	children = aux_c;

    // Open a log
    FILE *log = fopen("results/LOG", "w");

    fprintf(stderr, "Simulation in progress\n");
    for (i = 0; i < MAX_ITERATIONS; i++){
        fprintf(stderr, "\r%d", i);
        fflush(stderr);
        test();
	
        // Sort the retinas
        qsort(rs, NUM_INDIVIDUALS, sizeof(Retina), comparator);

        // Output stats
        for (j = 0; j < NUM_INDIVIDUALS; j++) fprintf(log, "%f ", rs[j].cost);
        fprintf(log, "\n");

        selection();
        crossover();
        mutation();
	    for (j = 0; j < NUM_INDIVIDUALS; j++){
                potentiate(&rs[j]);
            }
	}

    // Test and sort the retinas
    test();
    qsort(rs, NUM_INDIVIDUALS, sizeof(Retina), comparator);

	save(rs);

    fprintf(stderr, "\nDone. Removing trashes\n");

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        die(&rs[i]);
    }

	free_data();

	fclose(log);

	return 0;
}
