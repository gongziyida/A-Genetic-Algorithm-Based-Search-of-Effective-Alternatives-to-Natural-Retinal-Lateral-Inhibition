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
    TEST_SIZE, TRAIN_SIZE, T;
double TAU, ETA, NOISE, DICISION_BOUNDARY, XRATE;
bool INTERNAL_CONN;
std::string FOLDER;
Eigen::Matrix<double, 3, 1> W_COST;
Eigen::IOFormat TSV(4, Eigen::DontAlignCols, "\t", "\n", "", "", "", "");
// Precision, Alignment, Separators (elements, rows), Pre/Suffix (row, matrix)

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

void gaussian_filter(MatrixXd &signals, const MatrixXd &buffer, int n)
{
    const int r = (int)(CELLS * 0.1) - 1, len = r * 2 + 1;
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
		uniform(s1, 4, CELLS * 0.6 - 2);
        uniform(t1, s1 + 2, t1max);

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
    }
	gaussian_filter(signals, buffer, n);

    if (DICISION_BOUNDARY != 0)
    {
        MatrixXd labels = MatrixXd::Zero(n, num_sigs);
        labels.col(0) = st.col(T1) - st.col(S1);

        if (num_sigs == 2) labels.col(1) = st.col(T2) - st.col(S2);

        st = (labels.array() >= DICISION_BOUNDARY).cast<double>();
    }
}

double geq_prob(const MatrixXd &labels)
{
    return (labels.array() == 1).cast<double>().sum()
            / (labels.cols() * labels.rows());
}

double nn(const MatrixXd &x, const MatrixXd &y, MatrixXd &wih, MatrixXd &who,
          MatrixXd &hbias, MatrixXd &yhat, bool backprop = false)
{
    int n = x.rows(), in_features = x.cols(),
        h_features = wih.cols(), out_features = y.cols();

    MatrixXd h(n, h_features);
    h.noalias() = x * wih;

    MatrixXd mask(n, h_features);
    mask.noalias() = (h.array() >= 0).cast<double>().matrix(); // ReLU mask
    h = mask.array() * h.array();

    MatrixXd o(n, out_features);
    o.noalias() = (h * who).matrix();
    for (int i = 0; i < n; i++) o.row(i) += hbias;

    yhat.noalias() = (1 / (1 + exp(-o.array()))).matrix();

    MatrixXd res(n, out_features);
    res.noalias() = yhat - y;

    if (backprop)
    {
        MatrixXd delta(n, out_features);
        delta.noalias() = res / n;
        if (DICISION_BOUNDARY == 0)
            delta.array() *= yhat.array() * (1 - yhat.array());

        MatrixXd dwho(h_features, out_features);
        dwho.noalias() = h.transpose() * delta;

        MatrixXd delta_who_relu(n, h_features);
        delta_who_relu.noalias() = delta * who.transpose();
        delta_who_relu = delta_who_relu.array() * mask.array();

        MatrixXd dwih(in_features, h_features);
        dwih.noalias() = x.transpose() * delta_who_relu;

        MatrixXd dhbias(1, out_features);
        dhbias.noalias() = delta.colwise().sum();

        who.noalias() -= ETA * dwho;
        hbias -= ETA * dhbias;
        wih.noalias() -= ETA * dwih;
    }

    if (DICISION_BOUNDARY == 0) // MSE
        return res.array().pow(2).sum() / n / y.cols();
    else // BinaryCE
        return (-y.array() * log(yhat.array())
                - (1 - y.array()) * log(1 - yhat.array())).sum() / n;
}

inline double accuracy(const MatrixXd &yhat, const MatrixXd &y)
{
    int n = y.rows();
    MatrixXd yhat_bin(n, 1);
    yhat_bin.noalias() = (yhat.array() > 0.5).cast<double>().matrix();
    return (yhat_bin.array() == y.array()).cast<double>().sum() / n;
}

double model(double *auc, const MatrixXd &x, const MatrixXd &y, std::string *disp)
{
    int in_features = x.cols();
    int h_features = in_features / 2;
    MatrixXd wih = MatrixXd::Random(in_features, h_features);
    MatrixXd who = MatrixXd::Random(h_features, y.cols());
    MatrixXd hbias = MatrixXd::Random(1, y.cols());

    MatrixXd train_x = x.topRows(TRAIN_SIZE);
    MatrixXd train_y = y.topRows(TRAIN_SIZE);
    MatrixXd test_x = x.bottomRows(TEST_SIZE);
    MatrixXd test_y = y.bottomRows(TEST_SIZE);

    MatrixXd yhat(y.rows(), y.cols());
    MatrixXd loss(1, EPOCHS);
    MatrixXd acc(1, EPOCHS);

    for (int i = 0; i < EPOCHS; i++)
    {
        loss(i) = nn(train_x, train_y, wih, who, hbias, yhat, true);
        if (DICISION_BOUNDARY != 0)
            acc(i) = accuracy(yhat, train_y);
    }

    // AUC
    if (auc != NULL) *auc = loss.sum() / EPOCHS;

    if (disp != NULL)
    {
        std::stringstream ss_loss;
        ss_loss << "Loss: \n" << loss << "\n";
        *disp += ss_loss.str();

        if (DICISION_BOUNDARY != 0)
        {
            std::stringstream ss_acc;
            ss_acc << "Accuracy: \n" << acc << "\n";
            *disp += ss_acc.str();
        }
    }

    double ret = nn(test_x, test_y, wih, who, hbias, yhat);
    if (DICISION_BOUNDARY != 0)
        ret = accuracy(yhat, test_y);
    return ret;
}
