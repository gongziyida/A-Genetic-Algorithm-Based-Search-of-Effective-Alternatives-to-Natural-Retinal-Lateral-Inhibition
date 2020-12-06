#include <iostream>
#include <random>
#include <cmath>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "tool.h"
#define S1 0
#define T1 1
#define S2 2
#define T2 3

#define logit(x) (log(x / (1.0 - x)))
#define sigmoid(x) (1.0 / (1.0 + exp(-x)))

using Eigen::MatrixXd;

int THREADS, ITERS, POPULATION, ELITES, CELLS, EPOCHS,
    TEST_SIZE, TRAIN_SIZE, T;
double TAU, ETA, NOISE, DICISION_BOUNDARY, XRATE;
std::string FOLDER;
Eigen::IOFormat TSV(4, Eigen::DontAlignCols, "\t", "\n", "", "", "", "");
// Precision, Alignment, Separators (elements, rows), Pre/Suffix (row, matrix)

std::random_device sd;
std::mt19937 gen(sd());

double uniform(const double lo, const double hi)
{
	std::uniform_real_distribution<> dis(lo, hi);

    return dis(gen);
}

int uniform(const int lo, const int hi)
{
	std::uniform_int_distribution<> dis(lo, hi);

    return dis(gen);
}

void gaussian_filter(MatrixXd &signals, const MatrixXd &buffer, int n)
{
    const int r = (int)(CELLS * 0.05) - 1, len = r * 2 + 1;
    double s = 0.5 * r;
    s *= s;

    double filter[len];
    for (int x = -r, i = 0; x <= r; x++, i++)
        filter[i] = exp(- x * x / s / 2) / sqrt(2 * M_PI * s);

    for (int i = 0; i < n; i++)
    {
        double maxi = INT_MIN;
        double mini = INT_MAX;

        // Convolution & find min / max
        for (int j = 0; j < CELLS; j++)
        {
            for (int k = 0, x = j - r; k < len; k++, x++)
			{
				if (x < 0 || x >= CELLS) continue;
				signals(i, j) += filter[k] * buffer(i, x);
			}

            if (signals(i, j) > maxi) maxi = signals(i, j);
            if (signals(i, j) < mini) mini = signals(i, j);
        }
        // Normalize
        signals.row(i) = (signals.row(i).array() - mini) / (maxi - mini);
    }
}

void generate(MatrixXd &signals, MatrixXd &st, const int n, const int num_sigs)
{
    st = MatrixXd::Zero(n, num_sigs * 2);
    signals = MatrixXd::Zero(n, CELLS);
    MatrixXd buffer = MatrixXd::Random(n, CELLS) * NOISE;

    int t1max = (num_sigs == 1)? CELLS - 4 : CELLS * 0.8;

    for (int i = 0; i < n; i++)
    {
        // Randomly create two rectangles
        int s1, t1;
		s1 = uniform(4, (int) (CELLS * 0.6 - 3));
        t1 = uniform(s1 + 4, t1max - 1);

        buffer.block(i, s1, 1, t1 - s1).array() += 1;


        st(i, S1) = s1 / (CELLS + 0.01);
        st(i, T1) = t1 / (CELLS + 0.01);

        if (num_sigs == 2) // Treated as 1 if not 1 or 2
        {
            int s2, t2;
			s2 = uniform(s1, CELLS - 5);
            t2 = uniform(s2 + 2, CELLS - 3);

            buffer.block(i, s2, 1, t2 - s2).array() += 1;

            st(i, S2) = s2 / (CELLS + 0.01);
            st(i, T2) = t2 / (CELLS + 0.01);
        }
    }
	gaussian_filter(signals, buffer, n);

    for (int i = 0; i < n; i++)
    {
        double a = uniform(0.2, 1.0);
        double b = uniform(0.0, 1-a);

        signals.row(i) *= a; // scale
        signals.row(i).array() += b; // shift

        if (uniform(0, 99) < 50) signals.row(i).array() = 1 - signals.row(i).array(); // flip
    }

    if (DICISION_BOUNDARY != 0)
    {
        MatrixXd labels = MatrixXd::Zero(n, num_sigs);
        labels.col(0) = st.col(T1) - st.col(S1);

        if (num_sigs == 2) labels.col(1) = st.col(T2) - st.col(S2);

        st = (labels.array() >= DICISION_BOUNDARY).cast<double>();
    }
}

void generate(MatrixXd &signals, MatrixXd &x, const int n)
{
    x = MatrixXd::Zero(n, 1);
    signals = MatrixXd::Zero(n, CELLS);
    MatrixXd buffer = MatrixXd::Random(n, CELLS) * NOISE;

    for (int i = 0; i < n; i++)
    {
        // Randomly create two rectangles
        int s = uniform(4, CELLS - 5);

        buffer.block(i, s, 1, CELLS - s).array() += 1;


        x(i) = (double)s / CELLS;
    }
	gaussian_filter(signals, buffer, n);
}

double geq_prob(const MatrixXd &labels)
{
    return (labels.array() == 1).cast<double>().sum()
            / (labels.cols() * labels.rows());
}

double nn(const MatrixXd &x, const MatrixXd &y)
{
    int in_features = x.cols(), out_features = y.cols();
    int h_features = in_features / 4;

    // init
    MatrixXd wih = MatrixXd::Ones(in_features, h_features);
    MatrixXd who = MatrixXd::Ones(h_features, out_features);
    double hi = 1;// uniform(0.0, 1.0);
    double hh[out_features];

    for (int i = 0; i < out_features; i++)
        hh[i] = 1; //uniform(0.0, 1.0);

    for (int t = 0; t < EPOCHS + 1; t++)
    {
        int n = (t == EPOCHS)? TEST_SIZE : TRAIN_SIZE;

        // Input to hidden; before activation
        MatrixXd h_(n, h_features);
        if (t == EPOCHS)
            h_.array() = (x.bottomRows(n) * wih).array() + hi;
        else
            h_.array() = (x.topRows(n) * wih).array() + hi;

        // ReLU
        MatrixXd relu_mask(n, h_features);
        relu_mask.noalias() = (h_.array() >= 0).cast<double>().matrix();
        // Apply ReLU
        MatrixXd h(n, h_features);
        h.array() = relu_mask.array() * h_.array();

        // Hidden to output; before activation
        MatrixXd o_(n, out_features);
        o_.noalias() = h * who;
        for (int i = 0; i < out_features; i++)
            o_.col(i).array() += hh[i];

        // Sigmoid
        MatrixXd o(n, out_features);
        o.array() = 1 / (1 + exp(-o_.array()));


        // if (t != EPOCHS) // test cycle
        // if (t == EPOCHS)
        // {
            MatrixXd res(n, out_features);
            if (DICISION_BOUNDARY == 0) // MSE; Omega(0.1)
            {
                res.noalias() = o - y.bottomRows(n);

                double loss = res.array().pow(2).sum() / n / y.cols();
                std::cout << loss << ' '; // test
                // if (t == EPOCHS)
                // {
                    // std::cout << loss << '\n';
                    // return loss;
                // }
            }
            else // BCE; Omega(0.45)
            {
                res.array() = y.bottomRows(n).array() * log(o.array());
                res.array() += (1 - y.bottomRows(n).array()) * log(1 - o.array());
                double loss = -res.sum() / n;
                std::cout << loss << ' '; // test
                // if (t == EPOCHS)
                // {
                    // std::cout << loss << '\n';
                    // return loss;
                // }
            }
        // }

        // dE_do_
        MatrixXd delta(n, out_features);
        delta.noalias() = (o - y.topRows(n)) / n;
        if (DICISION_BOUNDARY == 0)
            delta.array() *= o.array() * (1 - o.array());


        MatrixXd delta_who_relu(n, h_features);
        delta_who_relu.noalias() = delta * who.transpose();
        delta_who_relu = delta_who_relu.array() * relu_mask.array();

        MatrixXd dwih(in_features, h_features);
        dwih.noalias() = x.topRows(n).transpose() * delta_who_relu;

        wih.noalias() -= ETA * dwih;

        who.noalias() -= ETA * h.transpose() * delta;

        for (int i = 0; i < out_features; i++)
            hh[i] -= ETA * delta.col(i).sum();
    }
    return 0;
}

double decoder(const MatrixXd &r, const MatrixXd &x, const MatrixXd &x0)
{
    MatrixXd denominator(r.rows(), 1);
    denominator.noalias() = r.rowwise().sum();
    MatrixXd out(r.rows(), 1);
    out.noalias() = r * x0;
    out = out.array() / denominator.array();
    out = out - x;
    return out.array().pow(2).sum() / r.rows();
}
