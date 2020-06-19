#ifndef RETINA_H
#define RETINA_H

#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <bits/stdc++.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;

#define MAX_TYPES 10
typedef std::bitset<32> binvec;

extern int LOSS, AUC, N_SYNAPSES;

class Retina;

struct Genome
{
	// Number of types of neurons
	int n_types;
	// Number of cells for each type
	int n_cell[MAX_TYPES];
	// Binary axon descriptors of interneuron types
    binvec axon[MAX_TYPES];
	// Binary dendrite descriptors of interneuron types
    binvec dendrite[MAX_TYPES];
	// Polarities of interneuron types
    double polarity[MAX_TYPES];
    // decay = exp(-((dist - beta) / phi)^2)
    double phi[MAX_TYPES];  // Scale, in [1, WIDTH)
    double beta[MAX_TYPES]; // Center, in [0, WIDTH)
	// Ganglion cell firing threshold
	double th;

	// Intervals between interneurons of the same types
    double intvl[MAX_TYPES];

	double i2e;
    Eigen::Matrix<double, 1, 3> costs; // Fitness cost, the larger the worse
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
	void react(const MatrixXd &in, MatrixXd &out);
	friend std::ostream & operator<<(std::ostream &os, const Retina &r);

private:
	int n; // Number of types
	double th; // Ganglion cell firing threshold
	int n_cell[MAX_TYPES];
	MatrixXd w[MAX_TYPES-1][MAX_TYPES];
};

#endif
