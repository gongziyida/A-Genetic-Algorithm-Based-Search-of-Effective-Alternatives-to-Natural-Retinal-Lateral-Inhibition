#include <iostream>
#include <fstream>
#include <thread>
#include <random>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "Retina.h"
#include "tool.h"
#include "GA.h"

// thread_local int TID;

void read_param()
{
    char aux[20];

    std::ifstream f("PARAM");
    if (f.is_open())
    {
        f >> aux >> THREADS >> aux >> ITERS >> aux >> POPULATION >> aux >> ELITES
          >> aux >> T >> aux >> TAU >> aux >> DT >> aux >> ETA >> aux >> EPOCHS
          >> aux >> CELLS >> aux >> RGCS
          >> aux >> NOISE >> aux >> TRAIN_SIZE >> aux >> TEST_SIZE >> aux >> TH
          >> aux >> W_COST(AUC) >> aux >> W_COST(N_SYNAPSES);

        f.close();

        W_COST(LOSS) = 1 - W_COST(AUC) - W_COST(N_SYNAPSES);
    } else
    {
        std::cout << "PARAM not found." << std::endl;
        std::exit(1);
    }
}

void test_reading()
{
    std::cout << THREADS << "\n" << ITERS << "\n" << POPULATION << "\n"
              << ELITES << "\n" << CELLS << "\n" << RGCS << "\n"
              << EPOCHS << "\n" << TEST_SIZE << "\n" << TRAIN_SIZE << "\n"
              << T << "\n" << TAU << "\n" << DT << "\n" << ETA << "\n" << NOISE
              << "\n" << TH << std::endl;
}

void write(Genome *g, const int tid)
{
    for (int i = 0; i < ELITES; i++)
    {
        std::string nameg = "results/" + std::to_string(tid) + "_"
                            + std::to_string(i) + "g.txt";
        std::ofstream fg(nameg);
        fg << g[i] << std::endl;
        fg.close();

        std::string namer = "results/" + std::to_string(tid) + "_"
                            + std::to_string(i) + "r.txt";
        std::ofstream fr(namer);
        fr << *g[i].r << std::endl;
        fr.close();
    }
}

void fork(int tid)
{
    MatrixXd sigs, st;

    generate(sigs, st, TRAIN_SIZE + TEST_SIZE, 1);
    if (TH != 0) std::cout << geq_prob(st) << std::endl;

    Genome g[POPULATION];
    Retina r[POPULATION];

    for (int i = 0; i < POPULATION; i++) r[i].init(g[i]);

    GA sim = GA(g, r);
    sim.run(sigs, st, tid);

    write(g, tid);
}

int main(int argc, char *argv[])
{
    read_param();
    // test_reading();

    std::string folder = (argc == 2)? argv[1] : "results";

    (void) std::system(("mkdir -p " + folder).c_str());
    (void) std::system(("rm " + folder + "/*").c_str());
    (void) std::system(("cp PARAM " + folder).c_str());

    std::thread ths[THREADS];
    for (int i = 0; i < THREADS; i++) ths[i] = std::thread(fork, i);
    for (int i = 0; i < THREADS; i++) ths[i].join();

    return 0;
}
