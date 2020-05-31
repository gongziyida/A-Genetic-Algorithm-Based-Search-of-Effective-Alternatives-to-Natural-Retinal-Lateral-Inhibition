#ifndef TOOL_H
#define TOOL_H

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
using Eigen::MatrixXd;

extern int THREADS, ITERS, POPULATION, ELITES, CELLS, RGCS, EPOCHS,
           TEST_SIZE, TRAIN_SIZE, T;
extern double TAU, ETA, NOISE, DICISION_BOUNDARY, XRATE;
extern bool INTERNAL_CONN;
extern Eigen::Matrix<double, 3, 1> W_COST;

void logitnormal(double &v, const double w, const double lo, const double hi);
void uniform(int &v, double lo, double hi);
void generate(MatrixXd &signals, MatrixXd &st, const int n, const int num_sigs);
double geq_prob(const MatrixXd &labels);
double model(double *auc, const MatrixXd &x, const MatrixXd &y, std::string *disp);

#endif
