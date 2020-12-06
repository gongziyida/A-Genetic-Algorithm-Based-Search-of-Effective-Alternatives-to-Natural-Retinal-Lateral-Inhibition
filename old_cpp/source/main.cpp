#include <iostream>
#include <fstream>
#include <thread>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "Retina.h"
#include "tool.h"
#include "GA.h"

// thread_local int TID;

void read_param(char *param)
{
    char aux[50];

    std::ifstream f(param);
    if (f.is_open())
    {
        f >> aux >> THREADS >> aux >> ITERS
          >> aux >> POPULATION >> aux >> ELITES
          >> aux >> XRATE >> aux >> CELLS
          >> aux >> T >> aux >> TAU >> aux >> ETA >> aux >> EPOCHS
          >> aux >> NOISE
          >> aux >> TRAIN_SIZE >> aux >> TEST_SIZE
          >> aux >> DICISION_BOUNDARY;
        f.close();

    } else
    {
        std::cout << "Parameters not found." << std::endl;
        std::exit(1);
    }
}

void test_reading()
{
    std::cout << THREADS << "\n" << ITERS << "\n" << POPULATION << "\n"
              << ELITES << "\n" << CELLS << "\n" << XRATE << "\n"
              << EPOCHS << "\n" << TEST_SIZE << "\n" << TRAIN_SIZE << "\n"
              << T << "\n" << TAU << "\n" << ETA << "\n" << NOISE
              << "\n" << DICISION_BOUNDARY << "\n" << std::endl;
}

void write(Genome *g, const int tid)
{
    for (int i = 0; i < ELITES; i++)
    {
        std::string nameg = FOLDER + "/" + std::to_string(tid) + "_"
                            + std::to_string(i) + "g.tsv";
        std::ofstream fg(nameg);
        fg << g[i] << std::endl;
        fg.close();

        std::string namer = FOLDER + "/" + std::to_string(tid) + "_"
                            + std::to_string(i) + "r.tsv";
        std::ofstream fr(namer);
        fr << *g[i].r << std::endl;
        fr.close();
    }
}

void fork(int tid)
{
    MatrixXd sigs, st;

    generate(sigs, st, TRAIN_SIZE + TEST_SIZE, 1);
    if (DICISION_BOUNDARY != 0) std::cout << geq_prob(st) << std::endl;

    Genome g[POPULATION];
    Retina r[POPULATION];

    GA sim = GA(g, r);
    sim.run(sigs, st, tid);

    write(g, tid);
}

// Testing main
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "./Simulation [folder] [parameters]" << std::endl;
        std::exit(1);
    }

    read_param(argv[2]);

    MatrixXd sigs, st;
    //
    // generate(sigs, st, 1, 1);
    // // std::cout << st.format(TSV) << '\n';
    // // std::cerr << sigs << '\n';
    //
    // int n = 1;
    //
    // Genome g[n];
    // Retina r[n];
    //
    // for (int i = 0; i < n; i++)
    // {
    //     g[i].r = &r[i];
    //     g[i].organize();
    //
    //     std::cout << g[i].n_types << std::endl;
    //
    //     g[i].r->init(g[i]);
    //     MatrixXd retina_out;
    //     g[i].r->react(sigs, retina_out, g[i]);
    //
    //     std::cerr << retina_out << '\n';
    //
    //     // std::cout << *g[i].r << std::endl;
    //     // std::cout << nn(retina_out, st) << std::endl;
    // }

    generate(sigs, st, 500, 1);
    std::cout << nn(sigs, st) << std::endl;
    return 0;
}

// int main(int argc, char *argv[])
// {
//     if (argc != 3)
//     {
//         std::cerr << "./Simulation [folder] [parameters]" << std::endl;
//         std::exit(1);
//     }
//
//     read_param(argv[2]);
//     // test_reading();
//
//     FOLDER = argv[1];
//
//     std::thread ths[THREADS];
//     for (int i = 0; i < THREADS; i++) ths[i] = std::thread(fork, i);
//     for (int i = 0; i < THREADS; i++) ths[i].join();
//
//     return 0;
// }
