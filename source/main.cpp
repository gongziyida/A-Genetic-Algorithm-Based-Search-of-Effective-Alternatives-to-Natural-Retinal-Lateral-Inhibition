#include <iostream>
#include <fstream>
#include "Retina.h"
#include "GA.h"

using namespace std;

int ITERS, POPULATION, ELITES, CELLS, RGCS, EPOCHS;
double T, TAU, DT, ETA, NOISE;

inline void read_param()
{
    char aux[20];

    ifstream f("PARAM");
    if (f.is_open())
    {
        f >> aux >> ITERS >> aux >> POPULATION >> aux >> ELITES
          >> aux >> T >> aux >> TAU >> aux >> DT >> aux >> ETA >> aux >> EPOCHS
          >> aux >> CELLS >> aux >> RGCS >> NOISE;

        cout << ITERS << " " << POPULATION << " " << ELITES << " "
             << T << " " << TAU << " " << DT << " " << CELLS << " " << RGCS << endl;
        f.close();
    } else cout << "PARAM not found." << endl;
}


int main()
{
    read_param();

    Genome g[10];
    Retina r[10];

    for (int i = 0; i < 10; i++)
    {
        r[i].init(g[i]);
    }

    return 0;
}
