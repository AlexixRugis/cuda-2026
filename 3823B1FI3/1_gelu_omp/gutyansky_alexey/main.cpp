#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

#include "gelu_omp.h"

using namespace std;

const int SZ = 134217728;

int main() {
    std::vector<float> input;
    input.resize(SZ, 0.5f);
    
    std::cout << GeluOMP(input)[0] << std::endl;

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    cout << 0.392405f / time << endl;
    cout << time << endl;
}