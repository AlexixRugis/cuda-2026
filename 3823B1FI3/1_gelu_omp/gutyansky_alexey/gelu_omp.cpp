// Compiler flags used: -O3 -mavx2 -ffast-math -fopenmp -flto

#pragma GCC optimize("Ofast")

#include "gelu_omp.h"

#include <vector>
#include <cmath>

template<typename T>
void unsafe_uninitialized_resize(std::vector<T>& v, size_t n) {
    struct Access : std::vector<T> {
        using std::vector<T>::_M_impl;
    };

    reinterpret_cast<Access&>(v)._M_impl._M_finish =
        reinterpret_cast<Access&>(v)._M_impl._M_start + n;
}

#pragma GCC target("avx2,fma")
std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int N = input.size();

    std::vector<float> result;
    result.reserve(N);
    unsafe_uninitialized_resize(result, N);

    const float _2SQRT2PI = 2.0f * sqrtf(2.0f / M_PI);
    const float C1 = 0.044715f;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float x = input[i];

        float x3 = x * x * x;
        float arg = _2SQRT2PI * (x + C1 * x3);
        float ex = expf(-arg);

        result[i] = x / (1.0f + ex);
    }


    return result;
}