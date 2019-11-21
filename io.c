// Author   Ziyi Gong

//#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
//#include "retina.h"
#include "io.h"

int MAX_ITERATIONS, NUM_INDIVIDUALS, NUM_ELITES, TRAIN_SIZE, TEST_SIZE,
    SIM_TIME, MAX_TYPES, MAX_CELLS, WIDTH;
double ETA;
double *TRAIN;
double *TEST;
int *LABELS_TR;
int *LABELS_TE;

char *PARAM = "PARAM";
char *DATA = "DATA";
char *LABELS = "LABELS";

char *PARAM_FORMAT =
        "Max Iterations = %d\n"
        "Num Individuals = %d\n"
        "Num Elites = %d\n"
        "Train Size = %d\n"
        "Test Size = %d\n"
        "Sim Time = %d\n"
        "Eta = %lf\n"
        "Max Types = %d\n"
        "Max Cells = %d\n"
        "Width = %d";

void load(){
    FILE *fparam = fopen(PARAM, "r");
    // Read parameters
    fscanf(fparam, PARAM_FORMAT, &MAX_ITERATIONS, &NUM_INDIVIDUALS, &NUM_ELITES,
            &TRAIN_SIZE, &TEST_SIZE, &SIM_TIME, &ETA, &MAX_TYPES, &MAX_CELLS, &WIDTH);

    fclose(fparam);

    FILE *fdata = fopen(DATA, "r");

    // Malloc
    TRAIN = mkl_malloc(TRAIN_SIZE * MAX_CELLS * sizeof(double), 64);
    LABELS_TR = mkl_malloc(TRAIN_SIZE * sizeof(int), 64);
    TEST = mkl_malloc(TEST_SIZE * MAX_CELLS * sizeof(double), 64);
    LABELS_TE = mkl_malloc(TEST_SIZE * sizeof(int), 64);

    // Read data
    int i, j;
    for (i = 0; i < TEST_SIZE + TRAIN_SIZE; i++){
        for (j = 0; j < MAX_CELLS; j++){
            if (i < TEST_SIZE){
                fscanf(fdata, "%lf%*c", &TEST[i*MAX_CELLS+j]);
            } else{
                fscanf(fdata, "%lf%*c", &TRAIN[(i-TEST_SIZE)*MAX_CELLS+j]);
            }
        }
    }

    fclose(fdata);

    FILE *flabels = fopen(LABELS, "r");

    // Read labels
    for (i = 0; i < TEST_SIZE + TRAIN_SIZE; i++){
        if (i < TEST_SIZE){
            fscanf(flabels, "%d%*c", &LABELS_TE[i]);
        } else{
            fscanf(flabels, "%d%*c", &LABELS_TR[i-TEST_SIZE]);
        }
    }

    fclose(flabels);
}

void free_data(){
    mkl_free(TRAIN);
    mkl_free(TEST);
    mkl_free(LABELS_TR);
    mkl_free(LABELS_TE);
}

void save(RetinaParam *rps){
    int nfrom, nto;
    char fname[20];
    FILE *f;

    for (int i = 0; i < NUM_ELITES; i++){
        snprintf(fname, 20, "results/%d.txt", i);

        f = fopen(fname, "w");
        fprintf(f, "%f\n", rps[i].cost);
        fprintf(f, "%d\n", rps[i].n_types);

        for (int j = 0; j < rps[i].n_connections; j++){
            if (rps[i].c[j].w == NULL) continue;

            nfrom = rps[i].n_cells[rps[i].c[j].from];
            nto = rps[i].n_cells[rps[i].c[j].to];

            if (nfrom == 0 || nto == 0) continue;

            fprintf(f, "# %d %d %d %d\n", rps[i].c[j].from, rps[i].c[j].to, nto, nfrom);

            for (int p = 0; p < nto; p++) {
                for (int q = 0; q < nfrom; q++) {
                    fprintf(f, "%.4f ", rps[i].c[j].w[p*nfrom+q]);
                }
                fprintf(f, "\n");
            }
        }
        fclose(f);
    }
}
