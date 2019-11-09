// Author   Ziyi Gong

#ifndef MY_PROJECT_IO_H
#define MY_PROJECT_IO_H
extern int MAX_ITERATIONS, NUM_INDIVIDUALS, NUM_ELITES, TRAIN_SIZE, TEST_SIZE, SIM_TIME,
            MAX_TYPES, MAX_CELLS, WIDTH;
extern  double ETA;
extern double *TRAIN;      // Training dataset
extern double *TEST;       // Testing dataset
extern int *LABELS_TR;  // Training labels
extern int *LABELS_TE;  // Testing labels

void load();
void free_data();
#endif //MY_PROJECT_IO_H




