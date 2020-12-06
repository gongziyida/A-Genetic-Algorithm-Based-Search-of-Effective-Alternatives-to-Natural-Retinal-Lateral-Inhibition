#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "Retina.h"
#include "tool.h"
#include "GA.h"

#define expth(x) (1.0e3 * exp((x - 4.0e4) / 1.0e3) - exp(-4.0e4 / 1.0e3))

using Eigen::MatrixXd;

inline int bernoulli(const int p1, const int p2)
{
    if (uniform(0, 99) < 50) return p1;
    return p2;
}

GA::GA(Genome *genomes, Retina *retinas)
{
    g = genomes;
    r = retinas;
    for (int i = 0; i < POPULATION; i++) g[i].r = &r[i];

    children = new Genome[POPULATION - ELITES];
    p1 = new int[POPULATION - ELITES];
    p2 = new int[POPULATION - ELITES];
}

void GA::eval(const MatrixXd &x, const MatrixXd &y)
{
    for (int i = 0; i < POPULATION; i++)
    {
        MatrixXd retina_out;
        g[i].r->react(x, retina_out, g[i]);

        double test_out = nn(retina_out, y);

        g[i].fit_cost = test_out; //(DICISION_BOUNDARY == 0)? test_out : (1 - test_out);
        // double cs = expth(g[i].n_synapses);
        // if (cs < 1e-3) cs = 0;
        g[i].total_cost = g[i].fit_cost;
    }
}


int comparator(const void *r1, const void *r2)
{
    /*
     * If r1's cost > r2's cost (r1 is woge), r1 ranks lower than r2. Return 1.
     * If r1's cost < r2's cost (r2 is woge), r1 ranks higher than r2. Return -1.
     * Else, return 0.
     */
    const Genome *r_a = (Genome *) r1;
    const Genome *r_b = (Genome *) r2;

    double a = r_a->total_cost, b = r_b->total_cost;

    if (a != a && b != b) return 0;
    else if (a != a) return 1;
    else if (b != b) return -1;

    if (a > b + 0.01)      return 1;
    else if (a < b - 0.01) return -1;
    else return (uniform(-1.0, 1.0) < 0);
}

int GA::select_p(const int p_)
{
    int p, rival1, rival2;
    do
    {// Randomly select two rivals
        rival1 = uniform(0, POPULATION - 1);
        rival2 = uniform(0, POPULATION - 1);
    } while (rival1 == rival2 || rival1 == p_ || rival2 == p_);

    double ratio1 = g[rival1].total_cost /
                    (g[rival1].total_cost + g[rival2].total_cost);

    p = (uniform(0, 99) < ratio1)? rival2 : rival1;

    return p;
}

void GA::selection()
{
    for (int i = 0; i < POPULATION - ELITES; i++)
    { // Choose each child's parents
        p1[i] = select_p(-1);
        p2[i] = select_p(p1[i]);
    }
}

void GA::crossover()
{
    // Crossover, results stored in buffer
    for (int i = 0; i < POPULATION - ELITES; i++)
    {
        int n = g[bernoulli(p1[i], p2[i])].n_types;
        children[i].n_types = n;

        int p, q;
        for (int j = 0; j < n; j++)
        {
            p = bernoulli(p1[i], p2[i]);

            if (j == 0)          q = 0; // Receptor cells
            else if (j == n - 1) q = g[p].n_types - 1; // Ganglion cells
            else
            { // Interneurons
                if (g[p].n_types == 2) // Another parent must have interneurons
                    p = (p == p1[i])? p2[i] : p1[i];

                q = uniform(1, g[p].n_types - 2);
            }

            children[i].axon[j] = g[p].axon[q];
            children[i].dendrite[j] = g[p].dendrite[q];
            // children[i].polarity[j] = g[p].polarity[q];
            children[i].n_cell[j] = g[p].n_cell[q];
            children[i].phi[j] = g[p].phi[q];
            children[i].beta[j] = g[p].beta[q];
            children[i].resistance[j] = g[p].resistance[q];
        }

        children[i].th = g[bernoulli(p1[i], p2[i])].th;
    }

    // Copy back
    for (int i = 0, k = ELITES; i < POPULATION - ELITES; i++, k++)
    {
        if (uniform(0, 99) > XRATE) continue; // crossover is binomial

        g[k].n_types = children[i].n_types;

        g[k].th = children[i].th;

        for (int j = 0; j < MAX_TYPES; j++)
        {
            g[k].axon[j] = children[i].axon[j];
            g[k].dendrite[j] = children[i].dendrite[j];
            // g[k].polarity[j] = children[i].polarity[j];
            g[k].n_cell[j] = children[i].n_cell[j];
            g[k].phi[j] = children[i].phi[j];
            g[k].beta[j] = children[i].beta[j];
            g[k].resistance[j] = children[i].resistance[j];
        }
    }
}
void mutate_single(double &var, const double lo, const double hi)
{
    double range = 0.01 * (hi - lo);
    double aux = uniform(-1, 1);
    var += aux * range;
    if (var < lo) var = lo;
    else if (var > hi) var = hi;
}

void GA::mutation()
{
    for (int i = ELITES; i < POPULATION; i++)
    {
        int n = g[i].n_types;
        // std::cout << i << " " << n << std::endl;

        for (int j = 0; j < n; j++)
        {
            // Mutate axon/dendrite, in very small amount per time
            mutate_single(g[i].axon[j], 0.0, M_PI * 2);
            mutate_single(g[i].dendrite[j], 0.0, M_PI * 2);

            // Mutate phi, in very small amount per time
            mutate_single(g[i].phi[j], 0.0, 0.5);

            // Mutate beta, in very small amount per time
            mutate_single(g[i].beta[j], -0.5, 0.5);

            // Mutate resistance, in very small amount per time
            if (j != 0) mutate_single(g[i].resistance[j], 0.0, 2.0);

            // 0.1 probability the polarity will flip
            // if (uniform(0, 99) < 10 && j > 0) g[i].polarity[j]  = -g[i].polarity[j];

            // Skip receptor and ganglion cells
            if (j == 0 || j == n - 1) continue;

            // 0.05 probability the number of cells will decrease/increase by 1
            if (uniform(0, 99) < 5)
            {
                int max_cell = CELLS;// (j == n - 1)? CELLS / 2 : CELLS;
                if (++g[i].n_cell[j] > CELLS) g[i].n_cell[j] = max_cell;
            } else if (uniform(0, 99) < 10)
            {
                int min_cell = 0;//(j == n - 1)? CELLS / 10 : 0;
                if (--g[i].n_cell[j] < min_cell) g[i].n_cell[j] = min_cell;
            }
        }

        // Mutate ganglion firing threshold, in very small amount per time
        mutate_single(g[i].th, 0.6, 1.0);

        // Force the receptog to have excitatory projections
        // if (g[i].polarity[0] < 0)
        //     g[i].polarity[0] = fabs(g[i].polarity[0]);
    }
}

void GA::start_competition(const MatrixXd &x, const MatrixXd &y)
{
    for (int j = 0; j < POPULATION; j++)
    {
        g[j].organize();
        g[j].r->init(g[j]);
    }

    eval(x, y);

    // Sort the retinas
    qsort(g, POPULATION, sizeof(Genome), comparator);
}

void GA::run(const MatrixXd &x, const MatrixXd &y, const int tid = 0)
{
    // Open a log
    std::ofstream f(FOLDER + "/" + "log" + std::to_string(tid) + ".tsv");

    for (int i = 0; i < ITERS; i++)
    {
        std::cout << "[" << tid << "]" << i + 1 << std::endl;

        start_competition(x, y);

        // Output stats
        for (int j = 0; j < POPULATION; j++)
        {
            f << g[j].fit_cost << "\t" << g[j].n_synapses << "\t"
              << "\t" << g[j].i2e << "\n";
        }
        f << "\n";

        selection();

        crossover();

        mutation();
	}

    // eval and sort the final retinas
    start_competition(x, y);

	f.close();

    delete[] children;
    delete[] p1;
    delete[] p2;
}
