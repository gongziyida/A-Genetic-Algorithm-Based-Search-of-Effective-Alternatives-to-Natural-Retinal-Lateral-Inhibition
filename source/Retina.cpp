#include <iostream>
#include <cmath>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "Retina.h"
#include "tool.h"
using Eigen::MatrixXd;

int LOSS = 0, AUC = 1, N_SYNAPSES = 2;

void Retina::init(Genome &g)
{
    n = g.n_types;
    th = g.th;

    for (int i = 0; i < n; i++) n_cell[i] = g.n_cell[i];

    // Make the weight from i to j
    for (int i = 0; i < n - 1; i++)
    { // ganglion cells do not project
        for (int j = 0; j < n; j++)
        {
            // internal connections, if allowed, only exist at i != 0 or n-1
            if (i == j && (!INTERNAL_CONN || i == 0 || i == n - 1))
                continue;

            int ni = n_cell[i];
            int nj = n_cell[j];

            if (ni == 0 || nj == 0) continue;

            // ganglion cells only get input from receptors
            if (j == n - 1 && i != 0) continue;

            w[i][j] = MatrixXd::Zero(ni, nj); // Take the ref

            // Calculate affinity between -1 and 1
            binvec buff = g.axon[j] ^ g.dendrite[i];
            double aff = 1.0 - buff.count() / 32.0;

            double maxi = INT_MIN;
            double mini = INT_MAX;

            // Calculate decay * affinity / distance
            for (int p = 0; p < ni; p++)
            {
                for (int q = 0; q < nj; q++)
                {
                    double d = fabs(g.intvl[i] * (p - ((double) ni - 1) / 2) -
                               g.intvl[j] * (q - ((double) nj - 1) / 2));

                    d = (d - g.beta[i]) / g.phi[i];
                    d = exp(-d * d);

                    double res = d * g.polarity[i] * aff;
                    w[i][j](p, q) = (res != res) ? 0 : res;

                    if (res > maxi) maxi = res;
                    if (res < mini) mini = res;
                }
            }

            // normalize and thresholding
            for (int p = 0; p < ni; p++)
            {
                for (int q = 0; q < nj; q++)
                {
                    double res = w[i][j](p, q) - (g.polarity[i] > 0 ? mini : maxi);
                    res /= (maxi - mini);

                    if (fabs(res) < 0.1) w[i][j](p, q) = 0;
                    else
                    {
                        g.costs(N_SYNAPSES) += 1;
                        w[i][j](p, q) = res;
                    }
                }
            }
        }
    }
}

inline void activation(const MatrixXd &a, MatrixXd &buffer)
{
    buffer.noalias() = (1 / (1 + exp(-a.array()))).matrix();
}

void Retina::react(const MatrixXd &in, MatrixXd &out)
{
    MatrixXd s_old[n];
    MatrixXd s_new[n];

    int r = in.rows();

    out = MatrixXd::Zero(r, n_cell[n-1]); // Clear

    for (int i = 0; i < n; i++)
    {
        s_old[i].noalias() = MatrixXd::Zero(r, n_cell[i]);
    }

    MatrixXd spikes(r, n_cell[n-1]);

    for (int t = 0; t < T; t++)
    {
        for (int i = 0; i < n; i++)
        {
            // V_i' = -V_i (+ I_ext)
            s_new[i].noalias() = -s_old[i];
            if (i == 0) s_new[i].noalias() += in;

            for (int j = 0; j < n - 1; j++)
            { // Ganglion cells do not project back
                if (j == i && (!INTERNAL_CONN || i == 0 || i == n - 1))
                    continue;

                // ganglion cells only get input from receptors
                if (i == n - 1 && j != 0) continue;

                MatrixXd buffer(s_old[j].rows(), s_old[j].cols());
                activation(s_old[j], buffer);
                s_new[i].noalias() += buffer * w[j][i];
            }

            // V_i = V_i + dt / tau * V_i'
            s_new[i] = s_old[i] + 1/TAU * s_new[i];
        }

        for (int i = 0; i < n; i++) // New becomes old
            s_old[i].noalias() = s_new[i];

        spikes.noalias() = (s_new[n-1].array() >= th).cast<double>().matrix();
        s_old[n-1].array() *= 1 - spikes.array(); // If fired, repolarize
        out.noalias() += spikes; // Ganglion firing
    }
    out.noalias() = out / (double)T; // firing rates

    // std::cout << s_old[n-1] << "\n********\n" << std::endl;
    // activation(s_old[n-1], out); // Activate ganglion

    // // Normalize output
	// MatrixXd o_max = out.rowwise().maxCoeff();
	// MatrixXd o_min = out.rowwise().minCoeff();
	// for (int i = 0; i < out.cols(); i++)
    //     out.col(i) = (out.col(i) - o_min).array() / (o_max - o_min).array();
}

std::ostream& operator<<(std::ostream &os, const Retina &r)
{
    for (int i = 0; i < r.n - 1; i++)
    {
        for (int j = 0; j < r.n; j++)
        {
            if (i == j && (!INTERNAL_CONN || i == 0 || i == r.n - 1))
                continue;
            if (j == r.n - 1 && i != 0) continue;

            os << "# " << i << "->" << j << " "
               << r.n_cell[i] << ":" << r.n_cell[j]
               << "\n" << r.w[i][j].format(TSV) << "\n";
        }
    }
    return os;
}

Genome::Genome()
{
    uniform(n_types, 2, MAX_TYPES);
    logitnormal((th = 0.5), 0.5, 0.1, 1);

    n_cell[0] = CELLS;

    for (int i = 0; i < n_types; i++)
    {
        if (i > 0) uniform(n_cell[i], 2, CELLS);

        int a, d;
        uniform(a, INT_MIN, INT_MAX);
        uniform(d, INT_MIN, INT_MAX);
        axon[i] = binvec(a);
        dendrite[i] = binvec(d);

        polarity[i] = beta[i] = 0;
        logitnormal(polarity[i], 1.5, -1, 1);
        logitnormal((phi[i] = 5), 3, 0.1, 10);
        logitnormal(beta[i], 1.5, -1, 1);
    }

    organize();
}

void Genome::organize()
{
    // Calculate intervals for receptors and ganglion cells
    intvl[0] = 1.0 / CELLS;
    intvl[n_types - 1] = 1.0 / n_cell[n_types - 1];
    costs *= 0;
    total_cost = 0;
    i2e = 1.0 / CELLS;

    if (n_types == 2) return;

    double inh = 0, exc = CELLS;

    // Check for void layers, i.e. with 0 cell or too small polarities
    for (int i = 1; i < n_types - 1; i++)
    {
        while (n_types > 2 &&
              (n_cell[i] == 0 ||
                  (polarity[i] > -0.01 && polarity[i] < 0.01)))
        {
            for (int j = i; j < n_types - 1; j++)
            {
                n_cell[j] = n_cell[j+1];
                axon[j] = axon[j+1];
                dendrite[j] = dendrite[j+1];
                polarity[j] = polarity[j+1];
                phi[j] = phi[j+1];
                beta[j] = beta[j+1];
            }
            n_types--;
        }
        intvl[i] = 1.0 / n_cell[i];

        if (polarity[i] < 0) inh += n_cell[i];
        else                 exc += n_cell[i];
    }

    i2e = inh / exc;
}

std::ostream & operator<<(std::ostream &os, const Genome &g)
{
    os << "n_types\tganglion_th\ttest_loss\tauc\tn_synapses\ttotal_cost\n";
    os << g.n_types << "\t" << g.th << "\t"
       << g.costs << "\t" << g.total_cost << "\n";

    os << "n_cell\taxon\tdendrite\tpolarity\tphi\tbeta\tintervals\n";

    for (int i = 0; i < g.n_types; i++)
    {
        os << g.n_cell[i] << "\t"
           << g.axon[i] << "\t" << g.dendrite[i] << "\t"
           << g.polarity[i] << "\t"
           << g.phi[i] << "\t" << g.beta[i] << "\t"
           << g.intvl[i] << "\n";
    }
    return os;
}
