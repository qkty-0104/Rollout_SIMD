#include <iostream>
#include <vector>
#include <algorithm>
#include <immintrin.h>  // SIMD
#include <ctime>
#include <limits>
#include <omp.h>        //OpenMP (简单测试，不作为本次实验主要内容)

using namespace std;

//Rollouts数目 
const int R = 100;

// 图内节点数目
const int N = 1000;

// 每条轨迹长度 （为与实际问题相符，设置长度上限而不是特定目的节点）
const int L = 100;

// 邻接矩阵
vector<vector<double>> graph(N, vector<double>(N));


// 随机初始化图（为与实际问题相符，不要求连通）
void init_graph() {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double weight = rand() / double(RAND_MAX);  // 连接或不连接
            graph[i][j] = weight;
            graph[j][i] = weight;
        }
    }
}

//// 随机初始化图（为与实际问题相符，不要求连通）
//void init_graph() {
//    srand(time(NULL));
//    __m256d rand_vec;
//    for (int i = 0; i < N; i++) {
//        rand_vec = _mm256_set_pd(rand() / double(RAND_MAX), rand() / double(RAND_MAX),
//            rand() / double(RAND_MAX), rand() / double(RAND_MAX));
//        for (int j = i + 1; j < N; j += 4) {
//            __m256d weight_vec = _mm256_loadu_pd(&graph[i][j]);
//            weight_vec = _mm256_blend_pd(weight_vec, rand_vec, 0b0001);
//            weight_vec = _mm256_blend_pd(weight_vec, rand_vec, 0b0010);
//            weight_vec = _mm256_blend_pd(weight_vec, rand_vec, 0b0100);
//            weight_vec = _mm256_blend_pd(weight_vec, rand_vec, 0b1000);
//            _mm256_storeu_pd(&graph[i][j], weight_vec);
//            _mm256_storeu_pd(&graph[j][i], weight_vec);
//        }
//    }
//}

// 基于Rollout算法生成轨迹并返回轨迹cost（128位）
double rollout_128(int start_node) {
    int current_node = start_node;
    double cost = 0.0;
    vector<int> trajectory(L);
    //#pragma omp parallel for reduction(+:cost)
    for (int i = 0; i < L; i++) {
        // 获取当前节点的邻居
        vector<double> neighbors(graph[current_node].begin(), graph[current_node].end());
        // 防止无限自环
        neighbors[current_node] = numeric_limits<double>::infinity();
        // 贪心策略寻找最近邻居（并行求解）
        __m128d min_neighbor = _mm_load_pd(&neighbors[0]);
        for (int j = 2; j < N; j += 2) {
            __m128d neighbor = _mm_load_pd(&neighbors[j]);
            min_neighbor = _mm_min_pd(min_neighbor, neighbor);
        }
        double min_cost;
        _mm_store_sd(&min_cost, min_neighbor);
        // 更新当前节点与轨迹cost
        int next_node = distance(neighbors.begin(), find(neighbors.begin(), neighbors.end(), min_cost));
        trajectory[i] = current_node;
        current_node = next_node;
        cost += min_cost;
        //if (current_node == start_node) {
        //    break;  // 回到初始节点则终止
        //}
    }
    //for (int i = 0; i < L; i++) {
    //    cout << trajectory[i] << "->";
    //}
    //cout << endl;
    return cost;
}


// 基于Rollout算法生成轨迹并返回轨迹cost（256位）
double rollout_256(int start_node) {
    int current_node = start_node;
    double cost = 0.0;
    vector<int> trajectory(L);
    //#pragma omp parallel for reduction(+:cost)
    for (int i = 0; i < L; i++) {
        // 获取当前节点的邻居
        vector<double> neighbors(graph[current_node].begin(), graph[current_node].end());
        // 防止无限自环
        neighbors[current_node] = numeric_limits<double>::infinity();
        // 贪心策略寻找最近邻居（并行求解）
        __m256d min_neighbor = _mm256_load_pd(&neighbors[0]);
        for (int j = 4; j < N; j += 4) {
            __m256d neighbor = _mm256_load_pd(&neighbors[j]);
            min_neighbor = _mm256_min_pd(min_neighbor, neighbor);
        }
        double min_cost;
        _mm_store_sd(&min_cost, _mm256_extractf128_pd(min_neighbor, 0));
        // 更新当前节点与轨迹cost
        int next_node = distance(neighbors.begin(), find(neighbors.begin(), neighbors.end(), min_cost));
        trajectory[i] = current_node;
        current_node = next_node;
        cost += min_cost;
        //if (current_node == start_node) {
        //    break;  // 回到初始节点则终止
        //}
    }
    //for (int i = 0; i < L; i++) {
    //    cout << trajectory[i] << "->";
    //}
    //cout << endl;
    return cost;
}

// 基于Rollout算法生成轨迹并返回轨迹cost（串行）
double rollout_serial(int start_node) {
    int current_node = start_node;
    double cost = 0.0;
    vector<int> trajectory(L);
    for (int i = 0; i < L; i++) {
        vector<double> neighbors(graph[current_node].begin(), graph[current_node].end());
        neighbors[current_node] = numeric_limits<double>::infinity();
        double min_cost = numeric_limits<double>::infinity();
        int next_node = current_node;
        for (int j = 0; j < N; j++) {
            if (j == current_node) continue;
            if (neighbors[j] < min_cost) {
                min_cost = neighbors[j];
                next_node = j;
            }
        }
        trajectory[i] = current_node;
        current_node = next_node;
        cost += min_cost;

        //if (current_node == start_node) {
        //    break;
        //}
    }
    //for (int i = 0; i < L; i++) {
    //    cout << trajectory[i] << "->";
    //}
    //cout << endl;
    return cost;
}

int main() {
    // initialize the graph
    init_graph();
    // generate 10 trajectories starting from different nodes
    clock_t start, end;

    start = clock();
    //#pragma omp parallel for
    for (int i = 0; i < R; i++) {
        int start_node = i % N;
        double cost = rollout_256(start_node);
        //cout << "Trajectory " << i + 1 << " starts from node " << start_node << " and has cost " << cost << endl;
    }
    end = clock();
    float total_time = (end - start) / float(CLOCKS_PER_SEC);
    cout << "mm256 Time " << total_time << endl;

    start = clock();
    //#pragma omp parallel for
    for (int i = 0; i < R; i++) {
        int start_node = i % N;
        double cost = rollout_128(start_node);
        //cout << "Trajectory " << i + 1 << " starts from node " << start_node << " and has cost " << cost << endl;
    }
    end = clock();
    total_time = (end - start) / float(CLOCKS_PER_SEC);
    cout << "mm128 Time " << total_time << endl;

    start = clock();
    for (int i = 0; i < R; i++) {
        int start_node = i % N;
        double cost = rollout_serial(start_node);
        //cout << "Trajectory " << i + 1 << " starts from node " << start_node << " and has cost " << cost << endl;
    }
    end = clock();
    total_time = (end - start) / float(CLOCKS_PER_SEC);
    cout << "Serial Time " << total_time << endl;

    return 0;
}
