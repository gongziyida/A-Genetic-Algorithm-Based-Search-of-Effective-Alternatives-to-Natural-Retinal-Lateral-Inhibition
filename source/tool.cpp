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
#define minmax(x, lo, hi) ((x - lo) / (hi - lo))
#define antiminmax(x, lo, hi) (x * (hi - lo) + lo)

using Eigen::MatrixXd;

int THREADS, ITERS, POPULATION, ELITES, CELLS, RGCS, EPOCHS,
    TEST_SIZE, TRAIN_SIZE;
double T, TAU, DT, ETA, NOISE;
Eigen::Matrix<double, 3, 1> W_COST;


void logitnormal(double &v, const double w, const double lo, const double hi)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::normal_distribution<> dis(0.0, 1.0);

	v = minmax(v, lo, hi);
	v = sigmoid(logit(v) + dis(gen) * w);
	v = antiminmax(v, lo, hi);
}

void uniform(int &v, const double lo, const double hi)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<> dis(0.0, 1.0);

	v = (int)std::round(antiminmax(dis(gen), lo, hi));
	if (v == hi) v--;
}

void generate(MatrixXd &signals, MatrixXd &st, const int n,
              const int num_sigs)
{
    st = MatrixXd::Zero(n, num_sigs * 2);
    signals = MatrixXd::Zero(n, CELLS);
    MatrixXd buffer = MatrixXd::Random(n, CELLS) * NOISE;
    double filter[7] = {0.065, 0.12, 0.175, 0.2, 0.175, 0.12, 0.065};

    for (int i = 0; i < n; i++)
    {
        // Randomly create two rectangles
        int s1, t1;
		uniform(s1, 2, CELLS * 0.7 - 2);
        uniform(t1, s1 + 2, CELLS * 0.8);

        buffer.block(i, s1, 1, t1 - s1).array() += 1;

        st(i, S1) = (double)s1 / CELLS;
        st(i, T1) = (double)t1 / CELLS;

        if (num_sigs == 2) // Treated as 1 if not 1 or 2
        {
            int s2, t2;
			uniform(s2, s1, CELLS - 4);
            uniform(t2, s2 + 2, CELLS - 2);

            buffer.block(i, s2, 1, t2 - s2).array() += 1;

            st(i, S2) = (double)s2 / CELLS;
            st(i, T2) = (double)t2 / CELLS;
        }

        double maxi = INT_MIN;
        double mini = INT_MAX;

        // Convolution & find min / max
        for (int j = 0; j < CELLS; j++)
        {
            for (int k = 0; k < 7; k++)
            {
                int l = j - k - 3;
                signals(i, j) += l < 0 ? 0 : filter[k] * buffer(i, l);
            }

            if (signals(i, j) > maxi) maxi = signals(i, j);
            if (signals(i, j) < mini) mini = signals(i, j);
        }
        // Normalize
        signals.row(i) = (signals.row(i).array() - mini) / (maxi - mini);
    }
}

double nn(const MatrixXd &x, const MatrixXd &y, MatrixXd &wih, MatrixXd &who,
          double &hbias, MatrixXd &yhat, bool backprop = false)
{
    int n = x.rows(), in_features = x.cols(),
        h_features = wih.cols(), out_features = y.cols();

    MatrixXd h(n, h_features);
    h.noalias() = x * wih;

    MatrixXd mask(n, h_features);
    mask.noalias() = (h.array() >= 0).cast<double>().matrix(); // ReLU mask
    h = mask.array() * h.array();

    MatrixXd o(n, out_features);
    o.noalias() = ((h * who).array() + hbias).matrix();

    yhat.noalias() = (1 / (1 + exp(-o.array()))).matrix();

    MatrixXd res(n, out_features);
    res.noalias() = yhat - y;

    if (backprop)
    {
        MatrixXd delta(n, out_features);
        delta.noalias() = (yhat.array() * (1 - yhat.array())
                           * res.array() / n / 4).matrix();

        MatrixXd drelu(h_features, h_features);
        drelu.noalias() = MatrixXd::Identity(h_features, h_features);

        double dhbias = delta.sum();
        MatrixXd dwih(in_features, h_features);
        dwih.noalias() = MatrixXd::Zero(in_features, h_features);
        MatrixXd dwho(h_features, out_features);
        dwho.noalias() = MatrixXd::Zero(h_features, out_features);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < h_features; j++) drelu(j, j) = mask(i, j);

            dwho.noalias() += h.row(i).transpose() * delta.row(i);
            dwih.noalias() += x.row(i).transpose() * delta.row(i)
                              * who.transpose() * drelu;
        }

        who.noalias() -= ETA * dwho;
        hbias -= ETA * dhbias;
        wih.noalias() -= ETA * dwih;
    }

    return res.array().pow(2).sum() / n / y.cols(); // MSE
}

double model(double *auc, const MatrixXd &x, const MatrixXd &y, std::string *disp)
{
    int in_features = x.cols();
    int h_features = in_features / 2;
    MatrixXd wih = MatrixXd::Random(in_features, h_features);
    MatrixXd who = MatrixXd::Random(h_features, y.cols());
    double hbias = (double)rand() / RAND_MAX;

    MatrixXd train_x = x.topRows(TRAIN_SIZE);
    MatrixXd train_y = y.topRows(TRAIN_SIZE);
    MatrixXd test_x = x.bottomRows(TEST_SIZE);
    MatrixXd test_y = y.bottomRows(TEST_SIZE);

    MatrixXd yhat;
    MatrixXd loss(1, EPOCHS);

    for (int i = 0; i < EPOCHS; i++)
        loss(i) = nn(train_x, train_y, wih, who, hbias, yhat, true);

    // normalized AUC
    if (auc != NULL) *auc = loss.sum() - EPOCHS * loss.minCoeff();

    if (disp != NULL)
    {
        std::stringstream ss;
        ss << loss << "\n";
        *disp += ss.str();
    }

    return nn(test_x, test_y, wih, who, hbias, yhat);
}
