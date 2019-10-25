// Author   Ziyi Gong
// Version  0.1
#ifndef RETINA_H
#define RETINA_H
extern const int N_FACTORS = 5;
extern const int WIDTH = 50;

typedef struct Connections{
    int a;      // type a
    int b;      // type b
    double *w;  // ij is the weight from j of type a to i of type b
} Connections;

typedef struct RetinaParam{
    double decay;       // Decadence of weight w.r.t. distance
    int n_types;        // Number of types of interneurons + one type of receptor
    double *axons;      // Axon descriptors of interneuron types
    double *dendrites;  // Dentrite descriptors of interneuron types
    int *n_cells;       // Number of cells for each type

    double *intvl;      // Intervals between interneurons of the same types
    int n_connections;  // Number of connection matrices
    Connections **c;    // Array of connection matrices

    double score;   // Fitness score, the larger the better
} RetinaParam;
#endif //MY_PROJECT_RETINA_H
