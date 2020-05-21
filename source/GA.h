#ifndef GA_H
#define GA_H

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "Retina.h"

class GA
{
public:
    GA(Genome *g, Retina *r);
    void run(const MatrixXd &x, const MatrixXd &y, const int tid);

private:
    int *p1, *p2;
    Genome *g, *children;
    Retina *r;

    void eval(const MatrixXd &x, const MatrixXd &y, bool disp);
    int select_p(const int cur, const int p_);
    void selection();
    void crossover();
    void mutation();
};

#endif
