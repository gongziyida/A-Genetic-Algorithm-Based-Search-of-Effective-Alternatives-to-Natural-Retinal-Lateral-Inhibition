#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include "Retina.h"
#include "tool.h"
#include "GA.h"

int DUP, ITERS, POPULATION, ELITES, CELLS, RGCS, EPOCHS,
    TEST_SIZE, TRAIN_SIZE;
double T, TAU, DT, ETA, NOISE;

void read_param()
{
    char aux[20];

    std::ifstream f("PARAM");
    if (f.is_open())
    {
        f >> aux >> DUP >> ITERS >> aux >> POPULATION >> aux >> ELITES
          >> aux >> T >> aux >> TAU >> aux >> DT >> aux >> ETA >> aux >> EPOCHS
          >> aux >> CELLS >> aux >> RGCS
          >> aux >> NOISE >> aux >> TRAIN_SIZE >> aux >> TEST_SIZE;

        // cout << ITERS << " " << POPULATION << " " << ELITES << " "
        //      << T << " " << TAU << " " << DT << " " << CELLS << " " << RGCS << std::endl;
        f.close();
    } else std::cout << "PARAM not found." << std::endl;
}

void write_genome(std::string &buffer, const Genome &g)
{
    buffer = std::to_string(i)
}

void write(Genome *g, Retina *r)
{
    for (int i = 0; i < ELITES; i++)
    {
        std::string nameg = "results/" + std::to_string(i) + "g.txt";
        std::ofstream fg(nameg);
        std::string genome;
        write_genome(genome, g[i]);
        fg << genome << std::endl;
        fg.close();

        std::string namer = "results/" + std::to_string(i) + "r.txt";
        std::ofstream fr(namer);

        fr << r[i] << std::endl;
        fr.close();
    }
}

int main()
{
    read_param();

    // TODO: Multi-thread

    MatrixXd sigs, st;
    generate(sigs, st, TRAIN_SIZE + TEST_SIZE, 1);

    Genome g[POPULATION];
    Retina r[POPULATION];

    for (int i = 0; i < POPULATION; i++)
    {
        std::cout << g[i] << std::endl;
        std::cout << r[i] << std::endl;
    }

    GA sim = GA(g, r);
    // sim.run(sigs, st);

    // write(g, r);

    return 0;
}
