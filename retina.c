// Author   Ziyi Gong
// Version  0.3
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl.h"
#include "retina.h"

void mk_connection(RetinaParam *rp){
    double affinityij, affinityji;
    int i, j, kij, kji, p, q;
    int n = rp->n_types;
    double decay = rp->decay;
    int ni, nj; // Number of cells
    double intvli, intvlj; // Intervals of i, j
    double d; // Distance-dependent weight factor
    int size; // Size to malloc

    // Make the weight from j to i
    for (i = 0; i < n - 1; i++){
        for (j = i + 1; j < n; j++){
            kij = i * n + j;
            kji = j * n + i;

            rp->c[kij].from = j;
            rp->c[kij].to = i;

            rp->c[kji].from = i;
            rp->c[kji].to = j;

            ni = rp->n_cells[i];
            nj = rp->n_cells[j];

            size = ni * nj * sizeof(double);

            rp->c[kij].w = malloc(size);
            rp->c[kji].w = malloc(size);

            // Calculate affinity between -1 and 1
            affinityij = cblas_ddot(N_FACTORS, &rp->axons[j * N_FACTORS], 1,
                                    &rp->dendrites[i * N_FACTORS], 1) / N_FACTORS;
            affinityji = cblas_ddot(N_FACTORS, &rp->axons[i * N_FACTORS], 1,
                                    &rp->dendrites[j * N_FACTORS], 1) / N_FACTORS;

            intvli = rp->intvl[i];
            intvlj = rp->intvl[j];

            // Calculate decay * affinity / distance
            for (p = 0; p < ni; p++){
                for (q = 0; q < nj; q++){
                    d = exp(- decay * abs(intvli * (p + 1) - intvlj * (q + 1)) / WIDTH);
                    if (d >= 1e-4){
                        rp->c[kij].w[p * nj + q] = d * affinityij;
                        rp->c[kji].w[q * ni + p] = d * affinityji;
                    } else{ // d < 1e-4 or is nan (overflow)
                        rp->c[kij].w[p * nj + q] = 0;
                        rp->c[kji].w[q * ni + p] = 0;
                    }
                }
            }
        }
    }
}

void mk_retina(RetinaParam *rp, int max_types, int max_cells){
    // Random stream init
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1);

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &rp->decay, 0, WIDTH);

    int n; // n_types
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &n, 2, max_types + 1);

    rp->n_types = n;
    rp->n_connections = n * n;

    rp->axons = malloc(n * N_FACTORS * sizeof(double));
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * N_FACTORS, rp->axons, -1, 1);

    rp->dendrites = malloc(n * N_FACTORS * sizeof(double));
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * N_FACTORS, rp->dendrites, -1, 1);

    rp->n_cells = malloc(n * sizeof(int));
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->n_cells, 1, max_cells);
    rp->n_cells[0] = max_cells;

    rp->intvl = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) rp->intvl[i] = (double) WIDTH / (rp->n_cells[i] + 1.0);

    rp->c = malloc(rp->n_connections * sizeof(Connections));
    mk_connection(rp);
}

void rm_retina(RetinaParam *rp){
    free(rp->axons);
    free(rp->dendrites);
    free(rp->n_cells);
    free(rp->intvl);

    for (int i = 0; i < rp->n_connections; i++) free(rp->c[i].w);
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
    mk_retina(rp, 5, 10);
    print_connections(rp, stdout);
    rm_retina(rp);
    free(rp);

    return 0;
}
