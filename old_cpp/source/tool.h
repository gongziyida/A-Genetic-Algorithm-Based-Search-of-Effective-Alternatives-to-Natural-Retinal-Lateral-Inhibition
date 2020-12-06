#ifndef TOOL_H
#define TOOL_H

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
using Eigen::MatrixXd;

extern int THREADS, ITERS, POPULATION, ELITES, CELLS, RGCS, EPOCHS,
           TEST_SIZE, TRAIN_SIZE, T;
extern double TAU, ETA, NOISE, DICISION_BOUNDARY, XRATE;
extern bool INTERNAL_CONN;
extern std::string FOLDER;
extern Eigen::Matrix<double, 3, 1> W_COST;
extern Eigen::IOFormat TSV;

double uniform(const double lo, const double hi);
int uniform(const int lo, const int hi);
void generate(MatrixXd &signals, MatrixXd &st, const int n, const int num_sigs);
void generate(MatrixXd &signals, MatrixXd &x, const int n);
double geq_prob(const MatrixXd &labels);
double nn(const MatrixXd &x, const MatrixXd &y);
double decoder(const MatrixXd &r, const MatrixXd &x, const MatrixXd &x0);

#endif
