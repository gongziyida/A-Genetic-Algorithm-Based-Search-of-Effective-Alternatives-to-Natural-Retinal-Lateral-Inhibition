// Author   Ziyi Gong
// Version  0.1
#ifndef RETINA_H
#define RETINA_H

typedef struct RetinaParam{
    // Subjective variables
    int width;      // Width & height of the scene/retina area
    double *scene;  // Visual scene whose size is the same as the retina area
    // Variables to optimize
    int n_types;             // Number of types of interneurons
    int *polarities;         // Polarities of interneuron types
    int *axons;              // Axon descriptors of interneuron types
    int *dendrites;          // Dentrite descriptors of interneuron types
    double *intvl;           // Intervals between interneurons of the same types
    double *rf_radii;        // Receptive field size of interneuron types
} RetinaParam;

typedef struct RetinaConnections{
    int n_types;
    double *w_coef;    // The ith-jth entry is the weight coefficient from neuron type j to type i.
    // TODO: Decide on how the processing is defined
} RetinaConnections;

#endif //MY_PROJECT_RETINA_H
