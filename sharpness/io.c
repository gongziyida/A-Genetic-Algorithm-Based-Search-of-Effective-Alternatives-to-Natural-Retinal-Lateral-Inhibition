// Author   Ziyi Gong

//#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
//#include "retina.h"
#include "io.h"

int MAX_ITERATIONS, NUM_INDIVIDUALS, NUM_ELITES, TRAIN_SIZE, TEST_SIZE,
    SIM_TIME, MAX_TYPES, MAX_CELLS, NUM_RGCS, WIDTH;
double TAU, DT, ETA;

char *PARAM = "PARAM";

char *PARAM_FORMAT =
        "Max Iterations = %d\n"
        "Num Individuals = %d\n"
        "Num Elites = %d\n"
        "Train Size = %d\n"
        "Test Size = %d\n"
        "Sim Time = %d\n"
        "Tau = %lf\n"
        "dt = %lf\n"
        "Eta = %lf\n"
        "Max Types = %d\n"
        "Max Cells = %d\n"
        "Num RGCs = %d\n"
        "Width = %d";

void load(){
    FILE *fparam = fopen(PARAM, "r");
    // Read parameters
    fscanf(fparam, PARAM_FORMAT, &MAX_ITERATIONS, &NUM_INDIVIDUALS, &NUM_ELITES,
            &TRAIN_SIZE, &TEST_SIZE, &SIM_TIME, &TAU, &DT, &ETA,
            &MAX_TYPES, &MAX_CELLS, &NUM_RGCS, &WIDTH);

    fclose(fparam);
}

void save(RetinaParam *rps){
    int ni, nj, n;
    char fname[20];
    FILE *f;

    int i, j, k;
    for (k = 0; k < NUM_ELITES; k++){
        snprintf(fname, 20, "results/%d.txt", k);

        n = rps[k].n_types;

        f = fopen(fname, "w");
        fprintf(f, "%f\n", rps[k].cost);
        fprintf(f, "%d\n", n);

        for (i = 0; i < n; i++){
            ni = rps[k].n_cells[i];

            if (ni == 0) continue;

            for (j = 0; j < n; j++){
                if (j == i) continue;

                nj = rps[k].n_cells[j];

                if (nj == 0) continue;

                fprintf(f, "# %d %d %d %d\n", j, i, nj, ni);

                for (int p = 0; p < ni; p++){
                    for (int q = 0; q < nj; q++) {
                        fprintf(f, "%.3f ", rps[k].c[j * n + i].w[p * ni + q]);
                    }
                    fprintf(f, "\n");
                }
            }
        }
        fclose(f);
    }
}
