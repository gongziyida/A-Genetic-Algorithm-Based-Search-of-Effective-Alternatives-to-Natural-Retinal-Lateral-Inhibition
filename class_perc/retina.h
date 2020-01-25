// Author   Ziyi Gong

#ifndef RETINA_H

#include <stdio.h>

#define RETINA_H

#define ABS_MAX_TYPES 10

typedef struct Layer{
	double *new_states;
	double *old_states;
	
	double *w[ABS_MAX_TYPES];
} Layer;

typedef struct Retina{
	// Number of types of neurons
	int n_types;
	// Number of cells for each type
	int n_cells[ABS_MAX_TYPES];       
	// Binary axon descriptors of interneuron types
    int axons[ABS_MAX_TYPES];       
	// Binary dendrite descriptors of interneuron types
    int dendrites[ABS_MAX_TYPES];     
	// Polarities of interneuron types
    double polarities[ABS_MAX_TYPES]; 
    // decay = exp(-((dist - beta) / phi)^2)
    double phi[ABS_MAX_TYPES];  // Scale, in [1, WIDTH)
    double beta[ABS_MAX_TYPES]; // Center, in [0, WIDTH)

	// Intervals between interneurons of the same types
    double intvl[ABS_MAX_TYPES];
    
	// Layers and projections
	Layer layers[ABS_MAX_TYPES];

    double avg_intvl;
    int n_synapses;
	int n_layers;
    double cost; // Fitness cost, the larger the worse
} Retina;

// Build the connection matrices
void potentiate(Retina *r);

// Initialize a retina; should be called for an appropriate number of times
void maker(Retina *r);

// Remove a retina; the method does not free the Retina itself
void die(Retina *r);

// Retinal process
void process(Retina *r, double *input, double *output);

#endif //MY_PROJECT_RETINA_H
