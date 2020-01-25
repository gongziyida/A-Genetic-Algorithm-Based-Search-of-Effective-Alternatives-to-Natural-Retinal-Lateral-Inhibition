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
double *TRAIN;
double *TEST;
int *SW_TR;
int *SW_TE;

char *PARAM = "PARAM";
char *DATA = "DATA";
char *SW = "SW";

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

    FILE *fdata = fopen(DATA, "r");

    // Malloc
    TRAIN = mkl_malloc(TRAIN_SIZE * MAX_CELLS * sizeof(double), 64);
    SW_TR = mkl_malloc(TRAIN_SIZE * sizeof(int), 64);
    TEST = mkl_malloc(TEST_SIZE * MAX_CELLS * sizeof(double), 64);
    SW_TE = mkl_malloc(TEST_SIZE * sizeof(int), 64);

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

    FILE *fsw = fopen(SW, "r");

    // Read labels
    for (i = 0; i < TEST_SIZE + TRAIN_SIZE; i++){
        if (i < TEST_SIZE){
            fscanf(fsw, "%d%*c", &SW_TE[i]);
        } else{
            fscanf(fsw, "%d%*c", &SW_TR[i-TEST_SIZE]);
        }
    }

    fclose(fsw);
}

void free_data(){
    mkl_free(TRAIN);
    mkl_free(TEST);
    mkl_free(SW_TR);
    mkl_free(SW_TE);
}

void save(Retina *rs){
    char fname[20];
    FILE *f;

	int k, i, j, p, q;
	int n_from, n_to;
	int n;

    for (k = 0; k < NUM_ELITES; k++){
        snprintf(fname, 20, "results/%d.txt", k);

        n = rs[k].n_types;

        f = fopen(fname, "w");
        fprintf(f, "%f\n", rs[k].cost);
        fprintf(f, "%d\n", n);

        for (i = 0; i < n; i++){ // from
			for (j = 0; j < n; j++){ // to
				if (j == i) continue;
				if (j == n - 1 && i != 0) continue; // interneurons to ganglion

				n_from = rs[k].n_cells[i];
				n_to = rs[k].n_cells[j];

				if (n_from == 0 || n_to == 0) continue; 

				fprintf(f, "# %d %d %d %d\n", i, j, n_to, n_from);

				for (p = 0; p < n_to; p++){
					for (q = 0; q < n_from; q++){
						fprintf(f, "%.3f ", 
								rs[k].layers[i].w[j][p * n_to + q]);
					}
					fprintf(f, "\n");
				}
			}

        }
        fclose(f);
    }
}
