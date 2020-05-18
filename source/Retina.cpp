#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "Retina.h"
#include "tool.h"
using Eigen::MatrixXd;

double affinity(Genome &g, int i, int j)
{
    double counter = 32.0;
    int buff = g.axon[j] ^ g.dendrite[i];
    while (buff != 0)
    {
        if (buff & 0x80000000) counter--;
        buff <<= 1;
    }
    return counter / 8 * sizeof(int);
}

void Retina::init(Genome &g)
{
    n = g.n_types;
    int i, j, p, q, ni, nj;
    double aff;
    double d; // Distance-dependent weight factor
    double mini, maxi;
    double res; // Auxiliary var to store the result

    g.n_synapses = 0;

    // Make the weight from i to j
    for (i = 0; i < n - 1; i++)
    { // ganglion cells do not project
        for (j = 0; j < n; j++)
        {
            if (i == j) continue;

            ni = n_cell[i];
            nj = n_cell[j];

            if (ni == 0 || nj == 0) continue;

            // ganglion cells only get input from receptors
            if (j == n - 1 && i != 0) continue;

            w[i][j] = MatrixXd::Zero(ni, nj); // Take the ref

            // Calculate affinity between -1 and 1
            aff = affinity(g, i, j);

            maxi = INT_MIN;
            mini = INT_MAX;

            // Calculate decay * affinity / distance
            for (p = 0; p < ni; p++)
            {
                for (q = 0; q < nj; q++)
                {
                    d = fabs(g.intvl[i] * (p - ((double) ni - 1) / 2) -
                            g.intvl[j] * (q - ((double) nj - 1) / 2));

                    d = (d - g.beta[i]) / g.phi[i];
                    d = exp(-d * d);

                    res = d * g.polarity[i] * aff;
                    w[i][j](p, q) = (res != res) ? 0 : res;

                    if (res > maxi) maxi = res;
                    if (res < mini) mini = res;
                }
            }

            // Normalize and thresholding
            for (p = 0; p < ni; p++)
            {
                for (q = 0; q < nj; q++)
                {
                    res = w[i][j](p, q) - (g.polarity[i] > 0 ? mini : maxi);
                    res /= (maxi - mini);

                    if (fabs(res) < 0.1) w[i][j](p, q) = 0;
                    else
                    {
                        g.n_synapses += 1;
                        w[i][j](p, q) = res;
                    }
                }
            }
        }
    }
}

inline void activation(const MatrixXd &a, MatrixXd &buffer)
{
    buffer = (1 / (1 + exp(-a.array()))).matrix();
}

void Retina::react(const MatrixXd &in, MatrixXd &out)
{
    int i, j, k;

    MatrixXd s_old[n];
    MatrixXd s_new[n];

    int r = in.rows();

    for (i = 0; i < n; i++)
    {
        s_old[i] = MatrixXd::Zero(r, n_cell[i]);
    }

    for (double t = 0; t < T; t += DT)
    {
        for (i = 0; i < n; i++)
        {
            if (n_cell[i] == 0) continue;

            // V_i' = -V_i (+ I_ext)
            s_new[i] = -s_old[i];
            if (i == 0) s_new[i] += in;

            for (j = 0; j < n - 1; j++)
            { // Ganglion cells do not project back
                if (j == i) continue;

                // ganglion cells only get input from receptors
                if (i == n - 1 && j != 0) continue;

                if (n_cell[j] == 0) continue;

                MatrixXd buffer;
                activation(s_old[j], buffer);
                s_new[i] += buffer * w[j][i];
            }

            // V_i = V_i + dt / tau * V_i'
            s_new[i] = s_old[i] + DT/TAU * s_new[i];
        }

        for (i = 0; i < n; i++)
        { // New becomes old
            s_old[i] = s_new[i];
        }
    }

    activation(s_old[n-1], out);
}

std::ostream& operator<<(std::ostream &os, const Retina &r)
{
    for (int i = 0; i < r.n - 1; i++)
    {
        for (int j = 0; j < r.n; j++)
        {
            if (i == j) continue;
            if (j == r.n - 1 && i != 0) continue;

            os << "# " << i << "->" << j << " "
               << r.n_cell[i] << ":" << r.n_cell[j]
               << "\n" << r.w[i][j] << "\n";
        }
    }
    return os;
}

Genome::Genome()
{
    uniform(n_types, 2, MAX_TYPES);
    for (int i = 1; i < n_types; i++)
    {
        uniform(n_cell[i], 2, CELLS);
        uniform(axon[i], INT_MIN, INT_MAX);
        uniform(dendrite[i], INT_MIN, INT_MAX);

        polarity[i] = phi[i] = beta[i] = 0;

        normal(polarity[i], 0.5, -1, 1);
        normal(phi[i], 0.5, -1, 1);
        normal(beta[i], 0.5, -1, 1);
    }

    organize();
}

void Genome::organize()
{
    intvl[0] = 1.0 / CELLS;
    intvl[n_types - 1] = 1.0 / n_cell[n_types - 1];
    n_synapses = 0;
    cost = 0;

    if (n_types == 2) return;

    for (int i = 1; i < n_types - 1; i++)
    {
        if (polarity[i] < 0.001 && polarity[i] > 0.001)
            polarity[i] = -0.001;

        while (n_cell[i] == 0)
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
    }
}

std::ostream & operator<<(std::ostream &os, const Genome &g)
{
    os << g.n_types << "\n";

    os << "n_cell,axon,dendrite,polarity,phi,beta,intervals,n_synapses,cost\n";

    for (int i = 0; i < g.n_types; i++)
    {
        os << g.n_cell[i] << ","
           << g.axon[i] << "," << g.dendrite[i] << ","
           << g.polarity[i] << ","
           << g.phi[i] << "," << g.beta[i] << ","
           << g.intvl[i] << "," << g.n_synapses << ","
           << g.cost << "\n";
    }
    return os;
}
