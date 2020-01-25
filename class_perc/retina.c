// Author   Ziyi Gong

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "mkl.h"
#include "retina.h"
#include "io.h"

VSLStreamStatePtr STREAM;

double affinity(Retina *r, int i, int j){
    double counter = 32.0;
    int buff = r->axons[j] ^ r->dendrites[i];
    while (buff != 0){
        if (buff & 0x80000000) counter--;
        buff <<= 1;
    }
    return counter / 8 * sizeof(int);
}

void potentiate(Retina *r){
    int i, j, p, q;
    int n = r->n_types;
    int ni, nj; // Number of cells
    double affinityij, affinityji;
//    double decay = r->decay;
    double d, dij, dji; // Distance-dependent weight factor
    double abs_maxij, abs_maxji; // Absolute max
    double res; // Auxiliary var to store the result

    r->avg_intvl = 0;
    r->n_layers = 0;
    r->n_synapses = 0;
    // Calculate the intervals first
    for (i = 0; i < n; i++){
        if (r->n_cells[i] > 0){
            r->n_layers++;
            r->intvl[i] = (double) WIDTH / r->n_cells[i];
        } else r->intvl[i] = WIDTH;
        r->avg_intvl += r->intvl[i];
    }

    r->avg_intvl /= n;

    // Make the weight from j to i
    for (i = 0; i < n - 1; i++){
        for (j = i + 1; j < n; j++){
            ni = r->n_cells[i];
            nj = r->n_cells[j];

            if (ni == 0 || nj == 0) continue;

            // ganglion cells only get input from receptors
            if (j == n - 1 && i != 0) continue; 
            
	    memset(r->layers[i].w[j], 0, nj * ni * sizeof(double));
            memset(r->layers[j].w[i], 0, nj * ni * sizeof(double));

            // Calculate affinity between -1 and 1
            affinityij = affinity(r, i, j);
            affinityji = affinity(r, j, i);

            abs_maxij = 0;
            abs_maxji = 0;

            // Calculate decay * affinity / distance
            for (p = 0; p < ni; p++){
                for (q = 0; q < nj; q++){
                    d = fabs(r->intvl[i] * (p - ((double) ni - 1) / 2) -
                            r->intvl[j] * (q - ((double) nj - 1) / 2));

                    dij = (d - r->beta[j]) / r->phi[j];
                    dij = exp(-dij * dij);

                    dji = (d - r->beta[i]) / r->phi[i];
                    dji = exp(-dji * dji);

//                    d = exp(-decay * fabs(r->intvl[i] * (p - 0.5 * (ni - 1)) -
//                            r->intvl[j] * (q - 0.5 * (nj - 1))) / WIDTH);

                    // Weights for cij
                    res = dij * r->polarities[j] * affinityij;
                    r->layers[j].w[i][p * nj + q] = isnan(res) ? 0 : res;

                    res = fabs(res);
                    if (res > abs_maxij) abs_maxij = res;

                    // Weights for cji
                    res = dji * r->polarities[i] * affinityji;
                    r->layers[i].w[j][q * ni + p] = isnan(res) ? 0 : res;

                    res = fabs(res);
                    if (res > abs_maxji) abs_maxji = res;
                }
            }

            // Normalize between -1 and 1
            if (abs_maxij > 0) cblas_dscal(ni * nj, 1/abs_maxij, r->layers[j].w[i], 1);
            if (abs_maxji > 0) cblas_dscal(ni * nj, 1/abs_maxji, r->layers[i].w[j], 1);


            for (p = 0; p < ni; p++) {
                for (q = 0; q < nj; q++) {
                    // Thresholding
                    if (fabs(r->layers[j].w[i][p * nj + q]) < 0.01) 
                        r->layers[j].w[i][p * nj + q] = 0;
                    else r->n_synapses += 1;

                    if (fabs(r->layers[i].w[j][q * ni + p]) < 0.01) 
                        r->layers[i].w[j][q * ni + p] = 0;
                    else r->n_synapses += 1;
                }
            }
        }
    }
}

void maker(Retina *r){
    if (STREAM == NULL)// Random STREAM init
        vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    int n; // n_types
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, 1, &n, 2, MAX_TYPES);

    r->n_types = n;

    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, r->axons, INT_MIN, INT_MAX);

    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, r->dendrites, INT_MIN, INT_MAX);

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, r->polarities, -1, 1);
    if (r->polarities[0] < 0) r->polarities[0] = fabs(r->polarities[0]);

    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, r->n_cells, 1, MAX_CELLS);
    r->n_cells[0] = MAX_CELLS;
    r->n_cells[n - 1] = NUM_RGCS;

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, r->phi, 1, WIDTH / 2);

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, r->beta, 0, WIDTH / 2);

    for (int i = 0; i < MAX_TYPES; i++){
        r->layers[i].new_states = mkl_malloc(MAX_CELLS * sizeof(double), 64);
        r->layers[i].old_states = mkl_malloc(MAX_CELLS * sizeof(double), 64);

        for (int j = 0; j < MAX_TYPES; j++){
            if (j == i) continue;

            r->layers[i].w[j] = mkl_malloc(MAX_CELLS * MAX_CELLS * 
                sizeof(double), 64);
        }        
    }

    potentiate(r);
}

void die(Retina *r){
    for (int i = 0; i < MAX_TYPES; i++){
        mkl_free(r->layers[i].new_states);
        mkl_free(r->layers[i].old_states);

        for (int j = 0; j < MAX_TYPES; j++){
            if (j == i) continue;

            mkl_free(r->layers[i].w[j]);
        }
    }
}

double activation(double val){
    return 1 / (1 + exp(-val));
}

void process(Retina *r, double *input, double *output){
    int n = r->n_types; // For convenience

    int i, j, k, ni, nj;
    double d[MAX_CELLS]; // For storing derivatives
    double o[MAX_CELLS]; // For storing output of a type
    double res[MAX_CELLS]; // For storing matrix-vector multiplication results

    /* For testing purose
    // Open a log
    FILE *log = fopen("results/TRACE", "w");
    // Output stats
    fprintf(log, "%d\n", n);
    */

    for (double t = 0; t < SIM_TIME; t+=DT){
        for (i = 0; i < n; i++) {
            ni = r->n_cells[i];

            if (ni == 0) continue;

            // Set the states to 0
            memset(r->layers[i].old_states, 0, MAX_CELLS * sizeof(double));
            memset(r->layers[i].new_states, 0, MAX_CELLS * sizeof(double));

            // V_i' = -V_i
            cblas_dcopy(ni, r->layers[i].old_states, 1, d, 1);
            cblas_dscal(ni, -1, d, 1);

            if (i == 0) // V_i' = -V_i + I_ext
                cblas_daxpy(ni, 1, input, 1, d, 1);

            for (j = 0; j < n; j++){
                if (j == i) continue;

                // ganglion cells only get input from receptors
                if (i == n - 1 && j != 0) continue; 
                

                nj = r->n_cells[j];

                if (nj == 0) continue;

                for (k = 0; k < nj; k++)
                    o[k] = activation(r->layers[j].old_states[k]);

                // I_j = W_ij * sigmoid(V_j)
                cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                            ni, nj, 1, 
                            r->layers[j].w[i], nj, 
                            o, 1, 0, res, 1);

                // V_i' = -V_i + sum_j(I_j) [+ I_ext]
                cblas_daxpy(ni, 1, res, 1, d, 1); 
            }

            /* For testing purose
            fprintf(stderr, "\n%d: %d\n", i, r->n_cells[i]);
            for (j = 0; j < MAX_CELLS; j++) {
                fprintf(stderr, "%f ", d[j]);
            }
             */

            // V_i = V_i + dt / tau * V_i'
            cblas_daxpy(ni, DT/TAU, d, 1, r->layers[i].new_states, 1);


        }

        /* For testing purose
        for (int k = 0; k < n; k++){
            for(int ki = 0; ki < r->n_cells[k]; ki++)
                fprintf(log, "%f ", r->new_states[k * MAX_CELLS + ki]);
            fprintf(log, "\n");
        }
        */

        for (i = 0; i < n; i++){ // New becomes old
            cblas_dcopy(MAX_CELLS, 
                        r->layers[i].new_states, 1, 
                        r->layers[i].old_states, 1);
        }
    }


    /* For testing purose
   fclose(log);
   */
    
    for (int i = 0; i < NUM_RGCS; i++){
        output[i] = activation(r->layers[n - 1].new_states[i]);
    }
}
