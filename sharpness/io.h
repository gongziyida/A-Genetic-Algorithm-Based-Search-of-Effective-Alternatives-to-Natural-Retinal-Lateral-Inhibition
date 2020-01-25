// Author   Ziyi Gong

#ifndef MY_PROJECT_IO_H

#include <stdio.h>
#include "retina.h"

#define MY_PROJECT_IO_H
extern int MAX_ITERATIONS, NUM_INDIVIDUALS, NUM_ELITES, TRAIN_SIZE, TEST_SIZE,
            SIM_TIME, MAX_TYPES, MAX_CELLS, NUM_RGCS, WIDTH;
extern double TAU, DT, ETA;

void load();
void save(RetinaParam *rps);
#endif //MY_PROJECT_IO_H




