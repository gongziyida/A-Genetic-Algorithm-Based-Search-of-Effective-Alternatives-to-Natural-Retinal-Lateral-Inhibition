#ifndef TOOL_H
#define TOOL_H

#include <Eigen/Dense>
using Eigen::MatrixXd;

extern int DUP, ITERS, POPULATION, ELITES, CELLS, RGCS, EPOCHS,
           TEST_SIZE, TRAIN_SIZE;
extern double T, TAU, DT, ETA, NOISE;

void normal(double &v, const double w, const double lo, const double hi);
void uniform(int &v, double lo, double hi);
void generate(MatrixXd &signals, MatrixXd &st, const int n,
              const int num_sigs);
double model(const MatrixXd &x, const MatrixXd &y);

#endif
