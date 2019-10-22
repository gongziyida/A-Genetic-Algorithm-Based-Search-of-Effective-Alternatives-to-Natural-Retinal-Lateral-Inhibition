// Author   Ziyi Gong
// Version  0.2
#include <stdio.h>
#include <stdlib.h>
#include <limit.h>
#include "mkl.h"
#include "retina.h"


void mk_retina(RetinaParam *rp, RetinaConnections *rc, width, double *scene,
                int max_types, int max_intvl, int max_rf_size){
    // Random stream init
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1);

    int n; // n_types
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &n, 1, max_types+1);

    rp->n_types = n;
    rc->n_types = n;
    rp->width = width;
    rp->scene = scene;

    rp->polarities = malloc(n * sizeof(int));
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->polarities, -1, 2);

    rp->axons = malloc(n * sizeof(int));
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->axons, INT_MIN, INT_MAX);

    rp->dendrites = malloc(n * sizeof(int));
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->dendrites, INT_MIN, INT_MAX);

    rp->intvl = malloc(n * sizeof(double));
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->intvl, 0, max_intvl);

    rp->rf_radii = malloc(n * sizeof(double));
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, rp->rf_radii, 0, max_rf_size);

    // Now start to make the weight coefficient matrix
    rc->w_coef = malloc(n * n * sizeof(double));

    int i, j, k, counter;
    unsigned int diff, mask;

    // Construct weight from j to i
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            k = i * n + j; // For "2D" array w_coef

            if (rp->polarities[j] == 0){ // Polarity == 0, no need to compute
                rc->w_coef[k] = 0;
                continue;
            }

            diff = (unsigned int) (rp->axons[j] ^ rp->dendrites[i]); // XOR

            // Count different bits
            mask = 0x1;
            counter = 0;
            do {
                if (diff & mask) counter++;
            } while ((mask <<=1) != 0);

            // The larger the counter, the larger the difference
            rc->w_coef[k] = (1 - counter / sizeof(unsigned int)) * rp->polarities[j];
        }
    }

    // TODO: Decide on how the processing is defined
}

void rm_retina(RetinaParam *rp, RetinaConnections *rc){
    free(rp->polarities);
    free(rp->axons);
    free(rp->ndrites);
    free(rp->intvl);
    free(rp->rf_radii);
    free(rc->w_coef);
}
