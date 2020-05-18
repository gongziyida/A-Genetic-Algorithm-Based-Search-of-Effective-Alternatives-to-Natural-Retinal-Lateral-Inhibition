#include <iostream>
#include <random>
#include <Eigen/Dense>
using Eigen::MatrixXd;

void normal(double &v, const double w, const double lo, const double hi)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::normal_distribution<> dis(0, 1);

	v += dis(gen) * w;

	if (v > hi) v = hi;
	else if (v < lo) v = lo;
}

void uniform(int &v, const double lo, const double hi)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_int_distribution<> dis(0, 1);

	v = (int)std::round(dis(gen) * (hi - lo) + lo);
}

void generate(MatrixXd &signals, MatrixXd &st, const int n,
              const int num_sigs)
{
    if (num_sigs != 1 && num_sigs != 2)
    {
        throw std::invalid_argument("only 1 or 2 is acceptable");
    }

    int i;

    st = MatrixXd::Zero(n, num_sigs * 2);
    signals = MatrixXd::Zero(n, CELLS + 1);
    MatrixXd buffer = MatrixXd::Random(n, CELLS) * NOISE;
    double filter[7] = {0.065, 0.12, 0.175, 0.2, 0.175, 0.12, 0.065};

    for (i = 0; i < n; i++)
    {
        // Randomly create two rectangles
        int s1 = uniform(2, CELLS * 0.7 - 2);
        int t1 = uniform(s1 + 2, CELLS * 0.8);

        buffer.block(i, s1, 1, t1 - s1).array() += 1;

        st(i, S1) = (double)s1 / CELLS;
        st(i, T1) = (double)t1 / CELLS;

        if (num_sigs == 2)
        {
            int s2 = uniform(s1, CELLS - 4);
            int t2 = uniform(s2 + 2, CELLS - 2);

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
    signals.col(CELLS).array() = 1;
}

double nn(const MatrixXd &x, const MatrixXd &y, MatrixXd &wih, MatrixXd &who,
          double &hbias, MatrixXd &yhat, bool backprop = false)
{
    int n = x.rows(), h_features = wih.cols();

    MatrixXd h = x * wih;

    MatrixXd mask = (h.array() >= 0).cast<double>(); // ReLU mask
    h = mask.array() * h.array();

    MatrixXd o = (h * who).array() + hbias;

    yhat = (1 / (1 + exp(-o.array())));

    MatrixXd res = yhat - y;

    if (backprop)
    {
        MatrixXd delta = yhat.array() * (1 - yhat.array()) * res.array() / n / 4;
        MatrixXd drelu = MatrixXd::Identity(h_features, h_features);

        double dhbias = delta.sum();
        MatrixXd dwho = MatrixXd::Zero(who.rows(), who.cols());
        MatrixXd dwih = MatrixXd::Zero(wih.rows(), wih.cols());

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < h_features; j++) drelu(j, j) = mask(i, j);

            dwho += h.row(i).transpose() * delta.row(i);
            dwih += x.row(i).transpose() * delta.row(i) * who.transpose() * drelu;
        }

        who -= ETA * dwho;
        hbias -= ETA * dhbias;
        wih -= ETA * dwih;
    }

    return res.array().pow(2).sum() / n / y.cols(); // MSE
}

double model(const MatrixXd &x, const MatrixXd &y)
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
    for (int i = 0; i < EPOCHS; i++)
    {
        cout << i << " " << nn(train_x, train_y, wih, who, hbias, yhat, true) << endl;
    }

    return nn(test_x, test_y, wih, who, hbias, yhat);
}
