#include <algorithm>
#include <random>
#include "Retina.h"

#define S1 0
#define T1 1
#define S2 2
#define T2 3


inline int uniform(double lo, double hi)
{
    double f = (double)rand() / RAND_MAX;
    return (int)(lo + f * (hi - lo));
}

void generate(MatrixXd &signals, MatrixXd &st, const int n)
{
    int i;

    st = MatrixXd::Zero(n, 4);
    signals = MatrixXd::Zero(n, CELLS + 1);
    MatrixXd buffer = MatrixXd::Random(n, CELLS) * NOISE;
    double filter[7] = {0.065, 0.12, 0.175, 0.2, 0.175, 0.12, 0.065};

    for (i = 0; i < n; i++)
    {
        // Randomly create two rectangles
        int s1 = uniform(2, CELLS * 0.7 - 2);
        int s2 = uniform(s1, CELLS - 4);
        int t1 = uniform(s1 + 2, CELLS * 0.8);
        int t2 = uniform(s2 + 2, CELLS - 2);

        buffer.block(i, s1, 1, t1 - s1).array() += 1;
        buffer.block(i, s2, 1, t2 - s2).array() += 1;

        st(i, S1) = (double)s1 / CELLS;
        st(i, T1) = (double)t1 / CELLS;
        st(i, S2) = (double)s2 / CELLS;
        st(i, T2) = (double)t2 / CELLS;

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

    return res.array().pow(2).sum() / n / 4; // MSE
}

double model(const MatrixXd &x, const MatrixXd &y)
{
    int in_features = x.cols();
    int h_features = in_features / 2;
    MatrixXd wih = MatrixXd::Random(in_features, h_features);
    MatrixXd who = MatrixXd::Random(h_features, y.cols());
    double hbias = 0.01;

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
void eval(const Genome &g[], const Retina &r[], const MatrixXd &x, const MatrixXd &y)
    for (int i = 0; i < POPULATION; i++){
        MatrixXd retina_out;
        r[i].react(x, retina_out);

        g[i].cost = model(retina_out, y);
    }
}


int comparator(const void *r1, const void *r2){
    /*
     * If r1's cost > r2's cost (r1 is worse), r1 ranks lower than r2. Return 1.
     * If r1's cost < r2's cost (r2 is worse), r1 ranks higher than r2. Return -1.
     * Else, return 0.
     */
    const Retina *rA = (Retina *) r1;
    const Retina *rB = (Retina *) r2;

    if (isnan(rA->cost)) return 1;
    if (isnan(rB->cost)) return -1;

    if (rA->cost > rB->cost) return 1;
    else if (rA->cost < rB->cost) return -1;
    else return 0;
}

int select_p(int cur, int another_p){
    int p, rival1, rival2;
    do {// Randomly select two rivals
        rival1 = rand() % NUM_INDIVIDUALS;
        rival2 = rand() % NUM_INDIVIDUALS;
    } while (rival1 == cur || rival2 == cur || rival1 == another_p || rival2 == another_p);

    double ratio = rs[rival1].cost / rs[rival2].cost;
    if (ratio > 1) ratio = 1.0 / ratio; // Assume the range is (0, 1]

    if (rand() % 100 < 100 - 50 * ratio){ // Pick the one with the lower cost (winner)
        p = (rs[rival1].cost < rs[rival2].cost)? rival1 : rival2;
    } else { // Pick the one with the high cost (loser)
        p = (rs[rival1].cost < rs[rival2].cost)? rival2 : rival1;
    }

    return p;
}

void selection(){
    for (int i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){ // Choose each child's parents
        p1[i] = select_p(i + NUM_ELITES, -1);
        p2[i] = select_p(i + NUM_ELITES, p1[i]);
    }
}

void crossover(){
    int p, n, q;
    int i, k;
    for (i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        if (rand() % 100 < 50)  p = p1[i];
        else                    p = p2[i];

        n = rs[p].n_types;

        for (k = 0; k < n; k++){
            if (rand() % 100 < 50)  p = p1[i];
            else                    p = p2[i];

            if (k == 0) // Receptor cells
                q = 0;
            else if (k == n - 1) // Ganglion cells
                q = rs[p].n_types - 1;
            else { // Interneurons
                if (rs[p].n_types > 2) // Has interneurons
                    q = 1 + rand() % (rs[p].n_types - 2);
                else {
                    p = p == p1[i] ? p2[i] : p1[i]; // Another parent must have interneurons
                    q = 1 + rand() % (rs[p].n_types - 2);
                }
            }

            children[i].axons[k] = rs[p].axons[q];
            children[i].dendrites[k] = rs[p].dendrites[q];
            children[i].polarities[k] = rs[p].polarities[q];
            children[i].n_cells[k] = rs[p].n_cells[q];
            children[i].phi[k] = rs[p].phi[q];
            children[i].beta[k] = rs[p].beta[q];
        }

//        children[i].decay = rs[p].decay;
        children[i].n_types = n;
    }

    for (i = 0; i < NUM_INDIVIDUALS - NUM_ELITES; i++){
        rs[NUM_ELITES + i].n_types = children[i].n_types;

        for (k = 0; k < MAX_TYPES; k++){
            rs[NUM_ELITES + i].axons[k] = children[i].axons[k];
            rs[NUM_ELITES + i].dendrites[k] = children[i].dendrites[k];
            rs[NUM_ELITES + i].polarities[k] = children[i].polarities[k];
            rs[NUM_ELITES + i].n_cells[k] = children[i].n_cells[k];
            rs[NUM_ELITES + i].phi[k] = children[i].phi[k];
            rs[NUM_ELITES + i].beta[k] = children[i].beta[k];
        }
    }
}


void mutation(){
    int n, i, j, chance, k;
    for (i = NUM_ELITES; i < NUM_INDIVIDUALS; i++){
        n = rs[i].n_types;

        // Randomly flip two bits for each axon and dendrite descriptor
        for (j = 0; j < n; j++){
            for (k = 0; k < 2; k++){
                rs[i].axons[j] ^= 1 << (rand() % 32);
                rs[i].dendrites[j] ^= 1 << (rand() % 32);
            }
        }

        for (j = 0; j < n; j++){
            // Mutate polarities, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rs[i].polarities[j], rs[i].polarities[j], 0.001);
            // Clip
            if (rs[i].polarities[j] > 1) rs[i].polarities[j] = 1;
            else if (rs[i].polarities[j] < -1) rs[i].polarities[j] = -1;
            else if (rs[i].polarities[j] == 0) rs[i].polarities[j] = 0.01; // in case of 0

            // Mutate phi, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rs[i].phi[j], rs[i].phi[j], 0.1);
            // Clip
            if (rs[i].phi[j] > WIDTH / 4) rs[i].phi[j] = WIDTH / 4;
            else if (rs[i].phi[j] < 1) rs[i].phi[j] = 1;

            // Mutate beta, in very small amount per time
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, STREAM,
                          1, &rs[i].beta[j], rs[i].beta[j], 0.1);
            // Clip
            if (rs[i].beta[j] > WIDTH / 2) rs[i].beta[j] = WIDTH / 2;
            else if (rs[i].beta[j] < 0) rs[i].beta[j] = 0;

            // Skip receptors and ganglion cells
            if (j == 0) continue;

            chance = rand() % 100;
            // 0.05 probability the number of cells will decrease/increase by 1
            if (chance < 5) {
                rs[i].n_cells[j] += 1;
                // Cannot go above the max
                if (rs[i].n_cells[j] > MAX_CELLS) rs[i].n_cells[j] = MAX_CELLS;
            } else if (chance < 10){
                rs[i].n_cells[j] -= 1;
                // Cannot go below the min (3)
                if (rs[i].n_cells[j] < 3) rs[i].n_cells[j] = 3;
            }
        }

        // Force the receptors to have excitatory projections
        // if (rs[i].polarities[0] < 0) rs[i].polarities[0] = fabs(rs[i].polarities[0]);
    }
}

int main(int argc, char **argv){
    fprintf(stderr, "Loading data\n");
    load();

    vslNewStream(&STREAM, VSL_BRNG_MT19937, 1);

    int i, j;

    fprintf(stderr, "Generating testing data\n");
    double aux_test[TEST_SIZE * MAX_CELLS];
    TEST = aux_test;
    double aux_sw[TEST_SIZE];
    SW = aux_sw;
    for (i = 0; i < TEST_SIZE; i++){
        SW[i] = generate(&TEST[i * MAX_CELLS]);
    }

    fprintf(stderr, "Initializing retinas\n");

    // Initialize individual space
	Retina aux_rs[NUM_INDIVIDUALS];
	rs = aux_rs;

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        maker(&rs[i]);
    }

    // Initialize parent index spaces
	int aux_p1[NUM_INDIVIDUALS], aux_p2[NUM_INDIVIDUALS];
	p1 = aux_p1;
    p2 = aux_p2;

	// Initialize children
	Retina aux_c[NUM_INDIVIDUALS - NUM_ELITES];
	children = aux_c;

    // Open a log
    FILE *log = fopen("results/LOG", "w");

    fprintf(stderr, "Simulation in progress\n");
    for (i = 0; i < MAX_ITERATIONS; i++){
        fprintf(stderr, "\r%d", i);
        fflush(stderr);
        test();

        // Sort the retinas
        qsort(rs, NUM_INDIVIDUALS, sizeof(Retina), comparator);

        // Output stats
        for (j = 0; j < NUM_INDIVIDUALS; j++) fprintf(log, "%f ", rs[j].cost);
        fprintf(log, "\n");

        selection();
        crossover();
        mutation();
	    for (j = 0; j < NUM_INDIVIDUALS; j++){
                potentiate(&rs[j]);
            }
	}

    // Test and sort the retinas
    test();
    qsort(rs, NUM_INDIVIDUALS, sizeof(Retina), comparator);

	save(rs);

    fprintf(stderr, "\nDone. Removing trashes\n");

    for (i = 0; i < NUM_INDIVIDUALS; i++){
        die(&rs[i]);
    }

	fclose(log);

	return 0;
}
