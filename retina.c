// Author   Ziyi Gong
// Version  0.3
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "mkl.h"
#include "retina.h"

double affinity(int i, int j){
    double counter = 32.0;
    int buff = rp->axons[j] ^ rp->dendrites[i];
    while (buff != 0){
        if (buff & 0x80000000) counter--;
        buff <<= 1;
    }
    return counter / 32;
}

void mk_connection(RetinaParam *rp){
    double affinityij, affinityji;
    int i, j, kij, kji, p, q;
    int n = rp->n_types;
    double decay = rp->decay;
    int ni, nj; // Number of cells
    double d; // Distance-dependent weight factor

    // Calculate the intervals first
    for (i = 0; i < n; i++)
        rp->intvl[i] = (double) WIDTH / (rp->n_cells[i] + 1.0);

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
            affinityij = affinity(i, j);
            affinityji = affinity(j, i);

            // Calculate decay * affinity / distance
            for (p = 0; p < ni; p++){
                for (q = 0; q < nj; q++){
                    d = exp(-decay * abs(rp->intvl[i] * (p + 1) - rp->intvl[j] * (q + 1)) / WIDTH);
                    if (d >= 1e-4){
                        rp->c[kij].w[p * nj + q] = d * rp->polarities[j] * affinityij;
                        rp->c[kji].w[q * ni + p] = d * rp->polarities[i] * affinityji;
                    } else{ // d < 1e-4 or is nan (overflow)
                        rp->c[kij].w[p * nj + q] = 0;
                        rp->c[kji].w[q * ni + p] = 0;
                    }
                }
            }
        }
    }
}

void init_retina(RetinaParam *rp){
    // Random stream init
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1);

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &rp->decay, 0, WIDTH);

    int n; // n_types
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &n, 2, MAX_TYPES + 1);

    rp->n_types = n;
    rp->n_connections = n * n;

    rp->axons = mkl_malloc(MAX_TYPES * sizeof(double), 64);
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * N_FACTORS, rp->axons, INT_MIN, INT_MAX);

    rp->dendrites = mkl_malloc(MAX_TYPES * sizeof(double), 64);
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * N_FACTORS, rp->dendrites, INT_MIN, INTMAX);

    rp->polarities = mkl_malloc(MAX_TYPES * sizeof(double), 64);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * N_FACTORS, rp->dendrites, -1, 1);

    rp->n_cells = mkl_malloc(MAX_TYPES * sizeof(int), 64);
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->n_cells, 1, MAX_CELLS);
    rp->n_cells[0] = MAX_CELLS;

    rp->states = mkl_malloc(2 * MAX_TYPES * MAX_CELLS * sizeof(double), 64);

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
    mkl_free(rp->polarities)
    mkl_free(rp->states)
    mkl_free(rp->n_cells);
    mkl_free(rp->intvl);

    for (int i = 0; i < rp->n_connections; i++) mkl_free(rp->c[i].w);
    free(rp->c);
}

void print_connections(RetinaParam *rp, FILE *f) {
    int nfrom, nto;
    for (int i = 0; i < rp->n_connections; i++) {
        if (rp->c[i].w == NULL) continue;

        nfrom = rp->n_cells[rp->c[i].from];
        nto = rp->n_cells[rp->c[i].to];

        fprintf(f, "%d -> %d (%d * %d)\n", rp->c[i].from,rp->c[i].to, nto, nfrom);

        for (int p = 0; p < nto; p++) {
            for (int q = 0; q < nfrom; q++) {
                fprintf(f, "%.4f ", rp->c[i].w[p * nfrom + q]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n------\n\n");
    }
}

int main(){
    RetinaParam *rp = malloc(sizeof(RetinaParam));
    init_retina(rp);
    print_connections(rp, stdout);
    rm_retina(rp);
    free(rp);

    return 0;
}
