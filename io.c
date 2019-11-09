// Author   Ziyi Gong

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "io.h"

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
        "Eta = %f\n"
        "Max Types = %d\n"
        "Max Cells = %d\n"
        "Width = %d"

void load(){
    FILE *fparam = fopen(PARAM, "r");
    // Read parameters
    fscanf(fparam, PARAM_FORMAT, &MAX_ITERATIONS, &NUM_INDIVIDUALS, &NUM_ELITES,
            &TEST_SIZE, &TRAIN_SIZE, &SIM_TIME, &ETA, &MAX_TYPES, &MAX_CELLS, &WIDTH);

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
                fscanf(fdata, "%f%*c", &TEST[i*MAX_CELLS+j]);
            } else{
                fscanf(fdata, "%f%*c", &TRAIN[(i-TEST_SIZE)*MAX_CELLS+j]);
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

    for (i = 0; i < TEST_SIZE + TRAIN_SIZE; i++){
        for (j = 0; j < MAX_CELLS; j++){
            if (i < TEST_SIZE){
                printf("%f\t", TEST[i*MAX_CELLS+j]);
            } else{
                printf("%f\t", TRAIN[(i-TEST_SIZE)*MAX_CELLS+j]);
            }
        }
        printf("\n");
        if (i < TEST_SIZE){
            printf("%d\n\n", LABELS_TE[i]);
        } else{
            printf("%d\n\n", LABELS_TR[i-TEST_SIZE]);
        }
    }
}

void free_data(){
    mkl_free(TRAIN);
    mkl_free(TEST);
    mkl_free(LABELS_TR);
    mkl_free(LABELS_TE);
}

int main(){
    load();
    free_data();
}