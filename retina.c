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

            rp->c[kij].a = i;
            rp->c[kij].b = j;

            rp->c[kji].a = j;
            rp->c[kji].b = i;

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
                        rp->c[kij].w[p * ni + q] = d * affinityij:
                        rp->c[kji].w[q * nj + p] = d * affinityji:
                    } else{ // d < 1e-4 or is nan (overflow)
                        rp->c[kij].w[p * ni + q] = 0:
                        rp->c[kji].w[q * nj + p] = 0:
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
    rp->scene = scene;

    rp->axons = malloc(n * sizeof(double));
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->axons, -1, 1);

    rp->dendrites = malloc(n * sizeof(double));
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->dendrites, -1, 1);

    rp->n_cells = malloc(n * sizeof(int));
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->intvl, 0, max_cells);

    rp->intvl = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) rp->intvl[i] = (double) WIDTH / (rp->n_cells[i] + 1.0);

    rp->n_connections = n * n;
    rp->c = malloc(rp->n_connections * (Connections));
    mk_connection(rp);
}

void rm_retina(RetinaParam *rp){
    free(rp->axons);
    free(rp->dendrites);
    free(rp->rf_radii);
    free(rp->n_cells);
    free(rp->intvl);

    for (int i = 0; i < rp->n_connections; i++) free(rp->c[i].w);
    free(rp->n_connections);
}

int main(){
    RetinaParam *rp = malloc(sizeof(RetinaParam));
    mk_retina(rp, 5, 10);
    rm_retina(rp);
    free(rp);

    return 0;
}
