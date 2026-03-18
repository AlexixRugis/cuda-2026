#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

#include "naive_gemm_omp.h"

using namespace std;

const int SZ = 1024;

int main() {
    std::vector<float> input_1;
    std::vector<float> input_2;
    input_1.resize(SZ * SZ);
    input_2.resize(SZ * SZ);
    
    std::mt19937 gen(10);
    uniform_real_distribution<float> distr(-1.0, 1.0f);

    for (auto& v : input_1) {
        v = distr(gen);
    }
    for (auto& v : input_2) {
        v = distr(gen);
    }
    
    std::cout << NaiveGemmOMP(input_1, input_2, SZ)[0] << std::endl;

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        NaiveGemmOMP(input_1, input_2, SZ);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    cout << 0.0541085f / time << endl;
    cout << time << endl;
}