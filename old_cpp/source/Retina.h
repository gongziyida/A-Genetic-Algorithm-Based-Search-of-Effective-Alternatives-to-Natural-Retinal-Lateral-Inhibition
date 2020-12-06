#ifndef RETINA_H
#define RETINA_H

#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;

#define MAX_TYPES 7

class Retina;

struct Genome
{
	// Number of types of neurons
	int n_types;
	// Number of cells for each type
	int n_cell[MAX_TYPES];
	// Binary axon descriptors of interneuron types
    double axon[MAX_TYPES];
	// Binary dendrite descriptors of interneuron types
    double dendrite[MAX_TYPES];
	// Polarities of interneuron types
    // int polarity[MAX_TYPES];
    // decay = exp(-((dist - beta) / phi)^2) * NOT USED
	// spatial selectivity: decay in (beta_+, beta_+ + phi)
    double phi[MAX_TYPES];  // Scale, in (0, 0.5]
    double beta[MAX_TYPES]; // Center, in [-0.5, 0.5]
	double resistance[MAX_TYPES]; // scale, in [0, 2]
	double th; // Ganglion cell firing threshold * HELD CONST NOW

	// Intervals between interneurons of the same types
    double intvl[MAX_TYPES];

	double i2e;
    double fit_cost; // Fitness cost, the larger the worse
	int n_synapses;
	double total_cost;

	Retina *r;

	Genome();
	void organize();
	friend std::ostream & operator<<(std::ostream &os, const Genome &g);
};

class Retina
{
public:
	void init(Genome &g);
	void react(const MatrixXd &in, MatrixXd &out, const Genome &g);
	friend std::ostream & operator<<(std::ostream &os, const Retina &r);

private:
	int n; // Number of types
	double th; // Ganglion cell firing threshold
	int n_cell[MAX_TYPES];
	MatrixXd w[MAX_TYPES-1][MAX_TYPES];
};

#endif
