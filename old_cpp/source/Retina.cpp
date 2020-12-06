#include <iostream>
#include <cmath>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "Retina.h"
#include "tool.h"
using Eigen::MatrixXd;

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
            // Feedforward only
            if (i == 0 && j == n - 1) continue;
            if (i != 0 && j != n - 1) continue;
            if (i == j) continue;

            // // internal connections only exist at i != 0 or i != n-1
            // if (i == j && (i == 0 || i == n - 1))
            //     continue;
            //
            // // ganglion cells only get input from receptors
            // if (j == n - 1 && i != 0) continue;

            int ni = n_cell[i];
            int nj = n_cell[j];

            if (ni == 0 || nj == 0) continue;

            w[i][j] = MatrixXd::Zero(ni, nj); // Take the ref

            // Calculate affinity between 0 and 1
            int aff = (cos(fabs(g.axon[i] - g.dendrite[j])) < 0)? -1 : 1;
            // aff = (aff > 0)? aff : 0;
            // if (aff == 0) continue;

            // double maxi = INT_MIN;
            // double mini = INT_MAX;

            // Calculate decay * affinity / distance
            for (int p = 0; p < ni; p++)
            {
                for (int q = 0; q < nj; q++)
                {
                    double d = fabs(g.intvl[i] * (p - ((double) ni - 1) / 2) -
                               g.intvl[j] * (q - ((double) nj - 1) / 2));

                    // d = (d - g.beta[i]) / g.phi[i];
                    // d = exp(-d * d) / g.n_cell[i];
                    double start = (g.beta[i] < 0)? 0 : g.beta[i];
                    double end = start + g.phi[i];
                    if (end > 1) end = 1;

                    // double res = d * g.polarity[i] * aff;
                    // double res = d;
                    // if ((res != res))// || res < g.resistance[i] / 2)
                    //     w[i][j](p, q) = 0;
                    // else
                    if (d >= start && d <= end)
                    {
                        w[i][j](p, q) = aff / (g.n_cell[i] * 2 * (end - start));
                        g.n_synapses += 1;
                    }

                    // if (res > maxi) maxi = res;
                    // if (res < mini) mini = res;
                }
            }

            // normalize and thresholding
            // for (int p = 0; p < ni; p++)
            // {
            //     for (int q = 0; q < nj; q++)
            //     {
            //         // double res = w[i][j](p, q) - (g.polarity[i] > 0 ? mini : maxi);
            //         // res /= (maxi - mini);
            //
            //         // if (uniform(0.0, maxi) < w[i][j](p, q))
            //         // {
            //             //     w[i][j](p, q) = 0.1 * g.polarity[i];
            //             //     g.n_synapses += 1;
            //             // }
            //             // else w[i][j](p, q) = 0;
            //
            //         if (fabs(w[i][j](p, q)) < 0.01) w[i][j](p, q) = 0;
            //         else
            //         {
            //             w[i][j](p, q) *= g.polarity[i] * 0.1;
            //
            //         }
            //     }
            // }
        }
    }
}

void Retina::react(const MatrixXd &in, MatrixXd &out, const Genome &g)
{
    MatrixXd s_old[n];
    MatrixXd s_new[n];

    int r = in.rows();

    out = MatrixXd::Zero(r, n_cell[n-1]); // Clear

    for (int i = 0; i < n; i++)
    {
        s_old[i].noalias() = MatrixXd::Ones(r, n_cell[i]) * 0.5;
    }

    // MatrixXd spikes(r, n_cell[n-1]);

    for (int t = 0; t < T; t++)
    {
        for (int i = 0; i < n; i++)
        {
            s_new[i].noalias() = MatrixXd::Zero(r, n_cell[i]);

            // From j to i
            for (int j = 0; j < n - 1; j++)
            {
                // Feedforward only
                if (j == 0 && i == n - 1) continue;
                if (j != 0 && i != n - 1) continue;
                if (i == j) continue;

                // // Receptors do not have internal connections
                // // Ganglion cells only receive from receptors
                // if ((i == 0 && j == 0) || (i == n - 1 && j != 0))
                //     continue;

                // If no connection exist
                // if (cos(fabs(g.axon[j] - g.dendrite[i])) <= 0) continue;

                // V_j * W_ji
                s_new[i].noalias() += s_old[j] * w[j][i];
            }
            if (i == 0) s_new[i].noalias() += in;

            // V_i' = [ I_in (+ I_ext) ] * exp(-V_i^2 / 0.2) * R
            s_new[i].noalias() = s_new[i] * g.resistance[i];

            // if (i != n - 1) // Non-spiking neurons has "bounds" on potentials
            // {
            //     // I_in * exp(-(V_i - 0.5)^2 / 0.2)
            //     s_new[i].array() *= exp(-5.0 * (s_old[i].array() - 0.5).pow(2));
            // }
            // else
            //     std::cerr << s_new[i] << '\n';

            // V_i' = -V_i + [ I_in (+ I_ext) ] * exp(-V_i^2 / 0.2) * R
            s_new[i].noalias() -= s_old[i];

            // V_i' = -V_i + [ I_in (+ I_ext) ] * exp(-V_i^2 / 0.2) * R + V_rest
            // V_rest is 0.5
            s_new[i].array() += 0.5;

            // V_i = V_i + dt / tau * V_i'
            s_new[i] = s_old[i] + 1.0/TAU * s_new[i];
        }

        for (int i = 0; i < n - 1; i++) // New becomes old
        {
            MatrixXd mask(s_new[i].rows(), s_new[i].cols());
            mask.noalias() = (s_new[i].array() > 0).cast<double>().matrix();
            s_old[i].array() = s_new[i].array() * mask.array();
            mask.noalias() = (s_old[i].array() > 1).cast<double>().matrix();
            s_old[i].array() = s_old[i].array() * (1 - mask.array()) + mask.array();
            // std::cout << s_old[i] << std::endl;
        }

        // Reset
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < n_cell[n-1]; j++)
            {
                s_old[n-1](i, j) = s_new[n-1](i, j); // New becomes old

                if (s_old[n-1](i, j) < th) continue;

                s_old[n-1](i, j) = -th / 2;

                // if (t >= T/2)
                out(i, j)++;
            }
        }

        // for (int i = 0; i < n; i++) // test
        // {
        //     std::cout << s_old[i] << std::endl;
        // }
    }
    out /= (double)T; // firing rates

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
            // Feedforward only
            if (i == 0 && j == r.n - 1) continue;
            if (i != 0 && j != r.n - 1) continue;
            if (i == j) continue;

            // if (i == j && (i == 0 || i == r.n - 1))
            //     continue;
            // if (j == r.n - 1 && i != 0) continue;

            os << "# " << i << "->" << j << " "
               << r.n_cell[i] << ":" << r.n_cell[j]
               << "\n" << r.w[i][j].format(TSV) << "\n";
        }
    }
    return os;
}

Genome::Genome()
{
    n_types = uniform(2, MAX_TYPES);

    th = 0.6;// uniform(0.6, 1.0);

    for (int i = 0; i < n_types; i++)
    {
        n_cell[i] = uniform(10, CELLS);

        // polarity[i] = (uniform(-1.1, 1.0) < 0)? -1 : 1;

        axon[i] = uniform(0.0, M_PI * 2);

        dendrite[i] = uniform(0.0, M_PI * 2);

        phi[i] = uniform(0.0, 0.5);

        beta[i] = uniform(-0.5, 0.5);

        resistance[i] = uniform(0.0, 2.0);
    }

    n_cell[0] = CELLS;
    n_cell[n_types - 1] = CELLS / 2;

    resistance[0] = 1;

    organize();
}

// // Test
// Genome::Genome()
// {
//     n_types = 4;
//
//     th = 0.6;// uniform(0.6, 1.0);
//
//     n_cell[0] = CELLS;
//     polarity[0] = 1;
//     phi[0] = 0.1;
//     beta[0] = 0.0;
//     resistance[0] = 1;
//
//     n_cell[1] = 25;
//     polarity[1] = -1;
//     phi[1] = 0.2;
//     beta[1] = 0.15;
//     resistance[1] = 1;
//
//     n_cell[2] = 25;
//     polarity[2] = 1;
//     phi[2] = 0.01;
//     beta[2] = 0.0;
//     resistance[2] = 1;
//
//     n_cell[3] = 25;
//     polarity[3] = -1;
//     phi[3] = 0.2;
//     beta[3] = 0.15;
//     resistance[3] = 1;
// }

void Genome::organize()
{
    // Calculate intervals for receptors and ganglion cells
    fit_cost = 0;
    n_synapses = 0;
    total_cost = 0;
    i2e = 1.0 / CELLS;

    if (n_types == 2) return;

    // double inh = 0, exc = CELLS;

    bool rm[n_types];

    // Check for void layers, i.e. with 0 cell / no connection
    for (int i = 1; i < n_types - 1; i++)
    {
        if (n_cell[i] == 0 || resistance[i] < 1e-3)
            rm[i] = true;
        else
            rm[i] = false;

        // int count_conn = 0;
        // for (int j = 1; j < n_types - 1; j++)
        // {
        //     if (i == j) continue;
        //     if (cos(fabs(axon[i] - dendrite[j])) > 0) count_conn++;
        // }
        //
        // if (count_conn == 0)
        //     rm[i] = true;
        // else
        //     rm[i] = false;
    }

    int n_aux = n_types;
    for (int i = n_types - 2; i > 0; i--)
    {
        if (!rm[i]) continue;

        for (int j = i; j < n_types - 1; j++)
        {
            n_cell[j] = n_cell[j+1];
            axon[j] = axon[j+1];
            dendrite[j] = dendrite[j+1];
            // polarity[j] = polarity[j+1];
            phi[j] = phi[j+1];
            beta[j] = beta[j+1];
        }
        n_aux--;
    }

    n_types = n_aux;


    for (int i = 0; i < n_types; i++)
    {
        intvl[i] = 1.0 / n_cell[i];
        //
        // if (polarity[i] < 0) inh += n_cell[i];
        // else                 exc += n_cell[i];
    }

    // i2e = inh / exc;
}

std::ostream & operator<<(std::ostream &os, const Genome &g)
{
    os << "n_types\tganglion_th\ttest_loss\tn_synapses\n";
    os << g.n_types << "\t" << g.th << "\t"
       << g.fit_cost << "\t" << g.n_synapses << "\n";

    os << "n_cell\taxon\tdendrite\tphi\tbeta\tinterval\tresistance\n";
    // os << "n_cell\tpolarity\tphi\tbeta\tinterval\tresistance\n";

    for (int i = 0; i < g.n_types; i++)
    {
        os << g.n_cell[i] << "\t"
           << g.axon[i] << "\t" << g.dendrite[i] << "\t"
           // << g.polarity[i] << "\t"
           << g.phi[i] << "\t" << g.beta[i] << '\t' << g.resistance[i] << "\t"
           << g.intvl[i] << "\n";
    }
    return os;
}
