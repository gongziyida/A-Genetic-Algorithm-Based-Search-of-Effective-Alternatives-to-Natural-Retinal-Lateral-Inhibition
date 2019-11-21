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

double affinity(RetinaParam *rp, int i, int j){
    double counter = 32.0;
    int buff = rp->axons[j] ^ rp->dendrites[i];
    while (buff != 0){
        if (buff & 0x80000000) counter--;
        buff <<= 1;
    }
    return counter / 8 * sizeof(int);
}

void mk_connection(RetinaParam *rp){
    double affinityij, affinityji;
    int i, j, kij, kji, p, q;
    int n = rp->n_types;
    double decay = rp->decay;
    int ni, nj; // Number of cells
    double d; // Distance-dependent weight factor

    rp->total_n_cells = 0;

    // Calculate the intervals first
    for (i = 0; i < n; i++){
        rp->intvl[i] = (double) WIDTH / (rp->n_cells[i] + 1.0);
        rp->total_n_cells += rp->n_cells[i];
    }


    // Make the weight from j to i
    for (i = 0; i < n - 1; i++){
        for (j = i + 1; j < n; j++){
            ni = rp->n_cells[i];
            nj = rp->n_cells[j];

            if (ni == 0 || nj == 0) continue;

            kij = j * n + i;
            kji = i * n + j;

            rp->c[kij].from = j;
            rp->c[kij].to = i;

            rp->c[kji].from = i;
            rp->c[kji].to = j;

            // Calculate affinity between -1 and 1
            affinityij = affinity(rp, i, j);
            affinityji = affinity(rp, j, i);

            // Calculate decay * affinity / distance
            for (p = 0; p < ni; p++){
                for (q = 0; q < nj; q++){
                    d = exp(-decay * abs(rp->intvl[i] * (p + 1) - rp->intvl[j] * (q + 1)) / WIDTH);
                    rp->c[kij].w[p * nj + q] = d * rp->polarities[j] * affinityij;
                    rp->c[kji].w[q * ni + p] = d * rp->polarities[i] * affinityji;

                    // Clip
                    if (rp->c[kij].w[p * nj + q] > 1)           rp->c[kij].w[p * nj + q] = 1;
                    else if (rp->c[kij].w[p * nj + q] < -1)     rp->c[kij].w[p * nj + q] = -1;
                    else if (fabs(rp->c[kij].w[p * nj + q]) < 0.01 || isnan(rp->c[kij].w[p * nj + q]))
                        rp->c[kij].w[p * nj + q] = 0;

                    if (rp->c[kji].w[q * ni + p] > 1)           rp->c[kji].w[q * ni + p] = 1;
                    else if (rp->c[kji].w[q * ni + p] < -1)     rp->c[kji].w[q * ni + p] = -1;
                    else if (fabs(rp->c[kji].w[q * ni + p]) < 0.01 || isnan(rp->c[kji].w[q * ni + p]))
                        rp->c[kji].w[q * ni + p] = 0;
                }
            }
        }
    }
}

void init_retina(RetinaParam *rp){
    if (STREAM == NULL)// Random STREAM init
        vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, 1, &rp->decay, 0, WIDTH);

    int n; // n_types
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, 1, &n, 2, MAX_TYPES + 1);

    rp->n_types = n;
    rp->n_connections = n * n;

    rp->axons = mkl_malloc(MAX_TYPES * sizeof(double), 64);
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, rp->axons, INT_MIN, INT_MAX);

    rp->dendrites = mkl_malloc(MAX_TYPES * sizeof(double), 64);
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, rp->dendrites, INT_MIN, INT_MAX);

    rp->polarities = mkl_malloc(MAX_TYPES * sizeof(double), 64);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, rp->polarities, -1, 1);

    rp->n_cells = mkl_malloc(MAX_TYPES * sizeof(int), 64);
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, STREAM, n, rp->n_cells, 1, MAX_CELLS);
    rp->n_cells[0] = MAX_CELLS;

    rp->new_states = mkl_malloc(MAX_TYPES * MAX_CELLS * sizeof(double), 64);
    rp->old_states = mkl_malloc(MAX_TYPES * MAX_CELLS * sizeof(double), 64);

    rp->intvl = mkl_malloc(MAX_TYPES * sizeof(double), 64);

    rp->c = malloc(MAX_TYPES * MAX_TYPES * sizeof(Connections));

    int i, j, kij, kji;
    int size = MAX_CELLS * MAX_CELLS * sizeof(double);
    for (i = 0; i < MAX_TYPES - 1; i++) {
        for (j = i + 1; j < MAX_TYPES; j++) {
            kij = i * n + j;
            kji = j * n + i;

            rp->c[kij].w = mkl_malloc(size, 64);
            rp->c[kji].w = mkl_malloc(size, 64);
        }
    }

    mk_connection(rp);
}

void rm_retina(RetinaParam *rp){
    mkl_free(rp->axons);
    mkl_free(rp->dendrites);
    mkl_free(rp->polarities);
    mkl_free(rp->new_states);
    mkl_free(rp->old_states);
    mkl_free(rp->n_cells);
    mkl_free(rp->intvl);

    for (int i = 0; i < rp->n_connections; i++) mkl_free(rp->c[i].w);
    free(rp->c);
}

void process(RetinaParam *rp, double *input) {
    int n = rp->n_types; // For convenience

    // Set the states of receptors to the input
    cblas_dcopy(MAX_CELLS, input, 1, rp->old_states, 1);

    // Set the rest of states to 0
    memset(&rp->old_states[MAX_CELLS], 0, (n - 1) * MAX_CELLS * sizeof(double));

    int a, b;
    double *buff;

    for (int t = 0; t < SIM_TIME; t++) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                a = rp->n_cells[i];
                b = rp->n_cells[j];

                if (a == 0 || b == 0) continue;

                // s_ki(t) += C_ij * s_kj(t-1)
                cblas_dgemv(CblasRowMajor, CblasNoTrans, a, b, 1, rp->c[j * n + i].w, b,
                            &rp->old_states[j * MAX_CELLS], 1, 1, &rp->new_states[i * MAX_CELLS],
                            1);

                // s_kj(t) += C_ji * s_ki(t-1)
                cblas_dgemv(CblasRowMajor, CblasNoTrans, b, a, 1, rp->c[i * n + j].w, a,
                            &rp->old_states[i * MAX_CELLS], 1, 1, &rp->new_states[j * MAX_CELLS],
                            1);
            }
        }

        if (t != SIM_TIME - 1) { // New becomes old
            buff = rp->old_states;
            rp->old_states = rp->new_states;
            rp->new_states = buff; // Need to switch reference in case of memory leaky
        } // Otherwise, we just keep the freshest values in new_states
    }
}