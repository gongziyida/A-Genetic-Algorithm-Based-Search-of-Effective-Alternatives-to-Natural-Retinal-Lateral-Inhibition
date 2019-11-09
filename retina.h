// Author   Ziyi Gong
// Version  0.1
#ifndef RETINA_H
#include <stdio.h>
#define RETINA_H
extern const int MAX_TYPES;
extern const int MAX_CELLS;
extern const int WIDTH;

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

    double *new_states; // The new states of nodes
    double *old_states; // The old states of nodes
    double *intvl;      // Intervals between interneurons of the same types
    int n_connections;  // Number of connection matrices
    Connections *c;     // Array of connection matrices
    /* Indexing table:
        s\t |   0   |   1   | ... | n-1
        ----|-------|-------|-----|---
          0 |   0   |   1   | ... | n-1
          1 |   n   |   1   | ... | 2n-1
          2 |  2n   | 2n+1  | ... | 3n-1
          ...
        n-1 | n^2-n |n^2-n+1| ... | n^2-1
     */

    double score;   // Fitness score, the larger the better
} RetinaParam;

// Build the connection matrices
void mk_connection(RetinaParam *rp);

// Initialize a retina; should be called for an appropriate number of times
void init_retina(RetinaParam *rp);

// Remove a retina; the method does not free the RetinaParam itself
void rm_retina(RetinaParam *rp);

// Print the connection matrices to a file
void print_connections(RetinaParam *rp, FILE *f);

#endif //MY_PROJECT_RETINA_H
