#ifndef GA_H
#define GA_H

#include <Eigen/Dense>
#include "Retina.h"

void generate(MatrixXd &signals, MatrixXd &st, const int n);
void eval(const Genome &g[], const Retina &r[], const MatrixXd &x, const MatrixXd &y);
int comparator(const void *g1, const void *g2);
void selection(const Genome &g[]);
void crossover(const Genome &g[]);
void mutation(const Genome &g[]);

#endif
