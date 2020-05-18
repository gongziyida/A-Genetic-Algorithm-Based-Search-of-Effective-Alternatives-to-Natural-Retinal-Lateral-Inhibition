#ifndef TOOL_H
#define TOOL_H

#include <Eigen/Dense>
using Eigen::MatrixXd;

void normal(double &v, const double w, const double lo, const double hi);
void uniform(int &v, double lo, double hi);
void generate(MatrixXd &signals, MatrixXd &st, const int n,
              const int num_sigs);
double model(const MatrixXd &x, const MatrixXd &y);

#endif
