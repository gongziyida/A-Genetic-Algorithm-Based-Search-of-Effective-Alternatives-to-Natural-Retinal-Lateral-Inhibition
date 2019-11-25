// Author   Ziyi Gong

#ifndef MY_PROJECT_IO_H

#include <stdio.h>
#include "retina.h"

#define MY_PROJECT_IO_H
extern int MAX_ITERATIONS, NUM_INDIVIDUALS, NUM_ELITES, TRAIN_SIZE, TEST_SIZE,
            SIM_TIME, MAX_TYPES, MAX_CELLS, WIDTH;
extern double TAU, ETA;
extern double *TRAIN;   // Training dataset
extern double *TEST;    // Testing dataset
extern int *LABELS_TR;  // Training labels
extern int *LABELS_TE;  // Testing labels

void load();
void free_data();
void save(RetinaParam *rps);
#endif //MY_PROJECT_IO_H




