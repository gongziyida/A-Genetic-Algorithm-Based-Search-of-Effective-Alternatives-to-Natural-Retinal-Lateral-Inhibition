#ifndef RETINA_H
#define RETINA_H

#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;

#define MAX_TYPES 10

struct Genome
{
	// Number of types of neurons
	int n_types;
	// Number of cells for each type
	int n_cell[MAX_TYPES];
	// Binary axon descriptors of interneuron types
    int axon[MAX_TYPES];
	// Binary dendrite descriptors of interneuron types
    int dendrite[MAX_TYPES];
	// Polarities of interneuron types
    double polarity[MAX_TYPES];
    // decay = exp(-((dist - beta) / phi)^2)
    double phi[MAX_TYPES];  // Scale, in [1, WIDTH)
    double beta[MAX_TYPES]; // Center, in [0, WIDTH)

	// Intervals between interneurons of the same types
    double intvl[MAX_TYPES];

    int n_synapses;
    double cost; // Fitness cost, the larger the worse

	Genome();
	void organize();
	friend std::ostream & operator<<(std::ostream &os, const Genome &g);
};

class Retina
{
public:
	void init(Genome &g);
	void react(const MatrixXd &in, MatrixXd &out);
	friend std::ostream & operator<<(std::ostream &os, const Retina &r);

private:
	int n; // Number of types
	int n_cell[MAX_TYPES];
	MatrixXd w[MAX_TYPES-1][MAX_TYPES];
};

#endif
