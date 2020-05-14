#include <cmath>
#include "Retina.h"

double Retina::affinity(Genome &g, int i, int j)
{
    double counter = 32.0;
    int buff = g.axons[j] ^ g.dendrites[i];
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

    g.avg_intvl = 0;
    g.n_layers = 0;
    g.n_synapses = 0;
    // Calculate the intervals first
    for (i = 0; i < n; i++)
    {
        n_cells[i] = g.n_cells[i];

        if (g.n_cells[i] > 0)
        {
            g.n_layers++;
            g.intvl[i] = (double) 1 / g.n_cells[i];
        }
        else
            g.intvl[i] = 1;

        g.avg_intvl += g.intvl[i];
    }

    g.avg_intvl /= n;

    // Make the weight from i to j
    for (i = 0; i < n - 1; i++)
    { // ganglion cells do not project
        for (j = 0; j < n; j++)
        {
            if (i == j) continue;

            ni = n_cells[i];
            nj = n_cells[j];

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

                    res = d * g.polarities[i] * aff;
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
                    res = w[i][j](p, q) - (g.polarities[i] > 0 ? mini : maxi);
                    res /= (maxi - mini);

                    if (fabs(res) < 0.1)
                        w[i][j](p, q) = 0;
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
        s_old[i] = MatrixXd::Zero(r, n_cells[i]);
    }

    for (double t = 0; t < T; t += DT)
    {
        for (i = 0; i < n; i++)
        {
            if (n_cells[i] == 0) continue;

            // V_i' = -V_i (+ I_ext)
            s_new[i] = -s_old[i];
            if (i == 0) s_new[i] += in;

            for (j = 0; j < n - 1; j++)
            { // Ganglion cells do not project back
                if (j == i) continue;

                // ganglion cells only get input from receptors
                if (i == n - 1 && j != 0) continue;

                if (n_cells[j] == 0) continue;

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
