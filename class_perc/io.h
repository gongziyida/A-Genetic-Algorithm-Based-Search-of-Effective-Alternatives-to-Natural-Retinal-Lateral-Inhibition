// Author   Ziyi Gong

#ifndef MY_PROJECT_IO_H

#include <stdio.h>
#include "retina.h"

#define MY_PROJECT_IO_H
extern int MAX_ITERATIONS, NUM_INDIVIDUALS, NUM_ELITES, TRAIN_SIZE, TEST_SIZE,
            SIM_TIME, MAX_TYPES, MAX_CELLS, NUM_RGCS, WIDTH;
extern double TAU, DT, ETA;
extern double *TRAIN;   // Training dataset
extern double *TEST;    // Testing dataset
extern int *SW_TR;  // Training labels
extern int *SW_TE;  // Testing labels

void load();
void free_data();
void save(Retina *rs);
#endif //MY_PROJECT_IO_H




