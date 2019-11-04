// Author   Ziyi Gong
// Version  0.1
#ifndef RETINA_H
#define RETINA_H
const int MAX_TYPES = 5;
const int MAX_CELLS = 10;
const int WIDTH = 50;

typedef struct Connections{
    int from;
    int to;
    double *w;  // ij is the weight from j of type "from" to i of type "to"
} Connections;

typedef struct RetinaParam{
    double decay;       // Decay of weight w.r.t. distance
    int n_types;        // Number of types of interneurons + one type of receptor
    int *axons;         // Binary axon descriptors of interneuron types
    int *dendrites;     // Binary dendrite descriptors of interneuron types
    double *polarities; // Polarities of interneuron types
    int *n_cells;       // Number of cells for each type

    double *states;     // The states of nodes, new followed by old (last time stamp)
    double *intvl;      // Intervals between interneurons of the same types
    int n_connections;  // Number of connection matrices
    Connections *c;     // Array of connection matrices

    double score;   // Fitness score, the larger the better
} RetinaParam;
#endif //MY_PROJECT_RETINA_H
