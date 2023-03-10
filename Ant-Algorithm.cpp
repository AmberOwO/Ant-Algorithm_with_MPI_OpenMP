#include <iostream>
#include <fstream>
#include <climits>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include <cstdlib> /* 亂數相關函數 */
#include <ctime>   /* 時間相關函數 */
#include <vector>
#include <utility>
#include <algorithm>
#include <iterator>
#include <mpi.h>
#include <omp.h>

using namespace std;

#define Q 100    // pheromone強度
#define Time 100    // 執行多少時間
#define ant_num 50 // m ants
#define thread_num 10

void Initialize_pheromone_matrix(int Map_size, vector<vector<double>> &Pheromone, vector<vector<int>> Map_vector);
void Place_on_random_cities(int start, int Map_size, vector<vector<int>> &kPath);
int Calculate_tour_distance(int city_num, vector<int> path, vector<vector<int>> Map);
int Roulette_wheel_selection(double alpha, double beta, double prob, int current_c, vector<int> City, vector<vector<double>> Pheromone, vector<vector<int>> Map);
void Update_pheromone_matrix(double rho, int Map_size, vector<vector<int>> kPath, vector<vector<double>> &Pheromone, vector<int> Lk);
double Sum_of_delta_Pheromone_ij(int i, int j, vector<vector<int>> kPath, vector<int> Lk);
void Read_File(string file_name, vector<vector<int>> &Map);

int main(int argc, char *argv[])
{
    // int Colonies = atoi(argv[1]);    // 幾個thread
    string infileName = argv[1]; // 讀取檔名
    double alpha = (double)atof(argv[2]); // pheromone factor
    double beta = (double)atof(argv[3]); // visibility factor
    double rho = (double)atof(argv[4]); // ρ

    int nprocess, id;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    double startwtime = 0.0, endwtime = 0.0; //記錄process開始時間、process結束時間
    double totalstart = 0.0, totalend = 0.0; //總執行時間
    startwtime = MPI_Wtime();

    int Map_size = 0;
    vector<vector<int>> Map_vector;
    if (id == 0)
    {
        Read_File(infileName, Map_vector);
        Map_size = Map_vector.size();
    }
    MPI_Bcast(&Map_size, 1, MPI_INT, 0, MPI_COMM_WORLD); // Bcast Map_size
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Bcast alpha
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Bcast beta
    MPI_Bcast(&rho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Bcast rho

    int *Map = new int[Map_size * Map_size]; // 1d array，用來MPI_Bcast Map
    if (id == 0)
    {
        for (int i = 0; i < Map_size; i++)
            for (int j = 0; j < Map_size; j++)
                Map[i * Map_size + j] = Map_vector[i][j];
    }
    MPI_Bcast(Map, Map_size * Map_size, MPI_INT, 0, MPI_COMM_WORLD); // Bcast Map

    if (id != 0) // 1d map to 2d
    {
        for (int i = 0; i < Map_size; i++)
        {
            vector<int> tmp;
            for (int j = 0; j < Map_size; j++)
                tmp.push_back(Map[i * Map_size + j]);
            Map_vector.push_back(tmp);
        }
    }

    int *TGlobal_all = new int [nprocess * (Map_size + 1)]; //二維陣列，計算過程中暫存
    int *LGlobal_all = new int [nprocess];
    int *recvcount = new int[nprocess];
    for (int i = 0; i < nprocess; ++i)
            recvcount[i] = Map_size + 1;
    int *displs = new int[nprocess];
    displs[0] = 0;
    for (int i = 1; i < nprocess; ++i)
            displs[i] = displs[i - 1] + (Map_size + 1);

    // Initialize TGlobal {this data is shared, everything else is private}
    
    int *TGlobal = new int[Map_size + 1];
    int LGlobal = INT_MAX;  

    #pragma omp parallel num_threads(thread_num)
    {
        // Initialize
        int tnum = omp_get_thread_num();
        int L = LGlobal;
        int *T = new int[Map_size + 1];
        vector<vector<int>> kPath(ant_num, vector<int>(Map_size + 1, 0));
        vector<int> Lk(ant_num, 0);

        // Initialize the pheromone matrix τ for each pair of cities
        vector<vector<double>> Pheromone; // pheromone matrix τ
        Initialize_pheromone_matrix(Map_size, Pheromone, Map_vector);

        // Place the m ants on n random cities
        random_device rd;  // non-deterministic generator
        mt19937 gen(rd() + id + tnum); // to seed mersenne twister.
        uniform_int_distribution<> dist(1, Map_size);
        for (int k = 0; k < ant_num; k++)
        {
            kPath[k][0] = dist(gen); // kPath[k][0] 存放起點
        }

        //prob
        uniform_real_distribution<> dist2(0, 1);
        

        // start loop
        for (int t = 0; t < Time; t++)
        {
            for (int k = 0; k < ant_num; k++)
            {
                int current_c = kPath[k][0];
                vector<int> City;
                for (int i = 1; i < Map_size + 1; i++)
                    if (i != kPath[k][0])
                        City.push_back(i);

                // Choose next city j according to the transition rule
                for (int i = 0; i < Map_size-1; i++)
                {
                    int current_tmp = current_c;
                    current_c = Roulette_wheel_selection(alpha, beta, dist2(gen), current_c, City, Pheromone, Map_vector);
                    kPath[k][current_tmp] = current_c;
                    vector<int> tmp;
                    for (int j = 0; j < City.size(); j++)
                    {
                        if (City[j] != current_c)
                            tmp.push_back(City[j]);
                    }

                    City.resize(City.size() - 1);
                    for (int j = 0; j < tmp.size(); j++)
                    {
                        City[j] = tmp[j];
                    }
                    tmp.clear();
                }
                kPath[k][current_c] = kPath[k][0];
                City.clear();
                City.shrink_to_fit();
            }

            for (int k = 0; k < ant_num; k++)
            {
                // Calculate tour distance Lk for ant k
                Lk[k] = Calculate_tour_distance(Map_size, kPath[k], Map_vector);

                // if an improved tour is found then
                if (Lk[k] < L)
                {
                    // Update T* and L*
                    L = Lk[k];
                    for (int i = 0; i < Map_size + 1; i++)
                    {
                        T[i] = kPath[k][i];
                    }
                }

                // if this is an exchange cycle then
                if ((t % 10 == 0) || (t == Time - 1))
                {
                    // if L* < LGlobal then
                    if (L < LGlobal)
                    {
                        #pragma omp critical
                        {
                            for (int i = 0; i < Map_size + 1; i++)
                            {
                                TGlobal[i] = T[i];
                            }
                            LGlobal = L;
                        }
                    }
                    #pragma omp barrier

                    for (int i = 0; i < Map_size + 1; i++)
                    {
                        T[i] = TGlobal[i];
                    }
                }
            }
            // Update the pheromone matrix τ
            Update_pheromone_matrix(rho, Map_size, kPath, Pheromone, Lk);
        }
        kPath.clear();
        kPath.shrink_to_fit();
    }

    // 回傳每個process的最佳值
    MPI_Gatherv(TGlobal, Map_size + 1, MPI_INT, TGlobal_all, recvcount, displs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&LGlobal, 1, MPI_INT, &LGlobal_all[id], 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (id == 0)
    {
        int *min = min_element( LGlobal_all, LGlobal_all + nprocess );
        int index = distance( LGlobal_all, min );

        
        cout << "\nTGlobal : \n";
        for (int i = 1; i < Map_size + 1; i++)
        {
            cout << TGlobal_all[index*(Map_size + 1)+i] - 1 << " ";
        }
        
        cout << "\nLGlobal = " << *min << "\n";
    }

    endwtime = MPI_Wtime();
    cout << "Process " << id << " execution time = " << endwtime - startwtime << endl;

    MPI_Finalize();
}

void Initialize_pheromone_matrix(int Map_size, vector<vector<double>> &Pheromone, vector<vector<int>> Map_vector)
{
    for (int i = 0; i < Map_size; i++)
    {
        vector<double> tmp;
        for (int j = 0; j < Map_size; j++)
        {
            if (i != j)
                tmp.push_back((double)1 / Map_vector[i][j]);

            else
                tmp.push_back(0);
        }
        Pheromone.push_back(tmp);
    }
}

void Place_on_random_cities(int start, int Map_size, vector<vector<int>> &kPath)
{
    /*
    random_device rd;  // non-deterministic generator
    mt19937 gen(rd() + id + tnum); // to seed mersenne twister.
    uniform_int_distribution<> dist(1, Map_size);
    */
   cout<<start<<" ";
    for (int k = 0; k < ant_num; k++)
        kPath[k][0] = start; // kPath[k][0] 存放起點
}

int Calculate_tour_distance(int city_num, vector<int> path, vector<vector<int>> Map) //
{
    int i = path[0], sum = 0;
    for (int c = 0; c < city_num; c++)
    {
        sum = sum + Map[i-1][path[i]-1];
        i = path[i];
    }
    return sum;
}

// 輪盤法
int Roulette_wheel_selection(double alpha, double beta, double prob, int current_c, vector<int> City, vector<vector<double>> Pheromone, vector<vector<int>> Map)
{
    if (City.size() == 1)
        return City[0];

    double sum = 0.0;
    vector<pair<int, double>> Pij;

    for (int i = 0; i < City.size(); i++)
    {
        int next_c = City[i];
        double a = pow((double)Pheromone[current_c - 1][next_c - 1], alpha);
        double b = pow((double)1 / (double)Map[current_c - 1][next_c - 1], beta);
        double fitness = a * b;
        pair<int, double> tmp (next_c,fitness);
        Pij.push_back(tmp);
        sum = sum + fitness;
    }

    for (int i = 0; i < City.size(); i++)
    {
        Pij[i].second = Pij[i].second / sum;
    }
    
    int select_city;
    double sum_prob = 0.0;
    
    for (int i = 0; i < City.size(); i++)
    {
        sum_prob = sum_prob + Pij[i].second;
        if (sum_prob >= prob)
        {
            select_city = Pij[i].first;
            break;
        }
    }
    Pij.clear();
    Pij.shrink_to_fit();
    return select_city;
}

// 更新pheromone matrix
void Update_pheromone_matrix(double rho, int Map_size, vector<vector<int>> kPath, vector<vector<double>> &Pheromone, vector<int> Lk)
{
    // τ[i][j] = (1 - ρ) * τ[i][j] + Sum_of_Δτij
    for (int i = 0; i < Map_size; i++)
    {
        for (int j = i + 1; j < Map_size; j++)
        {
            Pheromone[i][j] = (double)(1 - rho) * Pheromone[i][j] + Sum_of_delta_Pheromone_ij(i, j, kPath, Lk);
            //Pheromone[j][i] = Pheromone[i][j];
        }
    }
}

double Sum_of_delta_Pheromone_ij(int i, int j, vector<vector<int>> kPath, vector<int> Lk)
{
    double sum = 0;
    for (int k = 0; k < ant_num; k++)
    {
        /*if ant k use i to j*/
        if (kPath[k][i+1] == (j+1) || kPath[k][j+1] == (i+1))
            sum = sum + (double)Q / (double)Lk[k];
    }
    return sum;
}

void Read_File(string file_name, vector<vector<int>> &Map)
{
    ifstream fin(file_name);
    if (!fin)
        return;

    const char *d = "  ";
    string input;

    while (getline(fin, input)) // 讀一個檔案，並把tokens暫存在input_words
    {
        vector<int> tmp; // 暫存vector
        string num_str;

        for (char &x : input)
        {
            if (x != ' ' && x != '\0')
                num_str = num_str + x;

            else if (num_str != "")
            {
                tmp.push_back(stoi(num_str));
                num_str = "";
            }
        }
        if (num_str != "")
            tmp.push_back(stoi(num_str));
        Map.push_back(tmp);
    }
    fin.close();
}
