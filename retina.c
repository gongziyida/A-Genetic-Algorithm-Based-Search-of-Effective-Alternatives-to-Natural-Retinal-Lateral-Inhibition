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
    int i, j, kij, kji, p, q;
    int n = rp->n_types;
    int ni, nj; // Number of cells
    double affinityij, affinityji;
    double decay = rp->decay;
    double d; // Distance-dependent weight factor
    double abs_maxij, abs_maxji; // Absolute max
    double res; // Auxiliary var to store the result

    rp->avg_intvl = 0;
    rp->n_layers = 0;
    // Calculate the intervals first
    for (i = 0; i < n; i++){
        rp->intvl[i] = (double) WIDTH / (rp->n_cells[i] + 1.0);
        if (i > 0 && i < n - 1) rp->avg_intvl += rp->intvl[i];
        if (rp->n_cells[i] > 0) rp->n_layers++;
    }
    rp->avg_intvl /= n - 1;


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

            abs_maxij = 0;
            abs_maxji = 0;

            // Calculate decay * affinity / distance
            for (p = 0; p < ni; p++){
                for (q = 0; q < nj; q++){
                    d = exp(-decay * fabs(rp->intvl[i] * (p - 0.5 * (ni - 1)) -
                            rp->intvl[j] * (q - 0.5 * (nj - 1))) / WIDTH);

                    // Weights for cij
                    res = d * rp->polarities[j] * affinityij;
                    rp->c[kij].w[p * nj + q] = isnan(res) ? 0 : res;

                    res = fabs(res);
                    if (res < 0.05) rp->c[kij].w[p * nj + q] = 0; // Thresholding
                    if (res > abs_maxij) abs_maxij = res;

                    // Weights for cji
                    res = d * rp->polarities[i] * affinityji;
                    rp->c[kji].w[q * ni + p] = isnan(res) ? 0 : res;

                    res = fabs(res);
                    if (res < 0.05) rp->c[kij].w[q * ni + p] = 0; // Thresholding
                    if (res > abs_maxji) abs_maxji = res;
                }
            }

            // Normalize between -1 and 1
            if (abs_maxij > 0) cblas_dscal(ni * nj, 1/abs_maxij, rp->c[kij].w, 1);
            if (abs_maxji > 0) cblas_dscal(ni * nj, 1/abs_maxji, rp->c[kji].w, 1);
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
    rp->n_cells[n - 1] = MAX_CELLS / 5;

    rp->new_states = mkl_malloc(MAX_TYPES * MAX_CELLS * sizeof(double), 64);
    rp->old_states = mkl_malloc(MAX_TYPES * MAX_CELLS * sizeof(double), 64);

    rp->intvl = mkl_malloc(MAX_TYPES * sizeof(double), 64);

    rp->c = malloc(MAX_TYPES * MAX_TYPES * sizeof(Connections));

    int size = MAX_CELLS * MAX_CELLS * sizeof(double);
    for (int i = 0; i < MAX_TYPES * MAX_TYPES; i++) rp->c[i].w = mkl_malloc(size, 64);

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

    // Set the states to 0
    memset(rp->old_states, 0, MAX_TYPES * MAX_CELLS * sizeof(double));
    memset(rp->new_states, 0, MAX_TYPES * MAX_CELLS * sizeof(double));

    int t, i, j, ni, nj;
    double d[MAX_CELLS]; // For storing derivatives
    double res[MAX_CELLS]; // For storing matrix-vector multiplication results

    for (t = 0; t < SIM_TIME; t++){
        for (i = 0; i < n; i++) {
            ni = rp->n_cells[i];

            if (ni == 0) continue;

            // V_i' = -V_i
            cblas_dcopy(ni, &rp->old_states[i * MAX_CELLS], 1, d, 1);
            cblas_dscal(ni, -1, d, 1);

            if (i == 0) // V_i' = -V_i + I_ext
                cblas_daxpy(ni, 1, input, 1, d, 1);

            for (j = 0; j < n; j++){
                if (j == i) continue;

                nj = rp->n_cells[j];

                if (nj == 0) continue;

                // I_j = W_ji * V_j
                cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, nj, 1, rp->c[j * n + i].w, nj,
                            &rp->old_states[j * MAX_CELLS], 1, 0, res, 1);

                cblas_daxpy(ni, 1, res, 1, d, 1); // V_i' = -V_i + sum_j(I_j) [+ I_ext]
            }

            // V_i = V_i + dt / tau * V_i'
            cblas_daxpy(ni, 1/TAU, d, 1, &rp->new_states[i * MAX_CELLS], 1);

            for (j = 0; j < ni; j++) { // ReLU
                if (rp->new_states[i * MAX_CELLS + j] < 0)
                    rp->new_states[i * MAX_CELLS + j] = 0;
            }

        }

        if (t != SIM_TIME - 1) // New becomes old
            cblas_dcopy(MAX_TYPES * MAX_CELLS, rp->new_states, 1, rp->old_states, 1);
    }
}