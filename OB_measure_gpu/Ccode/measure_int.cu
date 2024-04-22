#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include "print.h"

using namespace std;

__device__ double klDivergence(const int *L1, const int *L2, int n, int maxdata) {
    double klDiv = 0.0;
    for (int i = 0; i < maxdata; ++i) {
        double P = (double) L1[i] / n;
        double Q = (P + (double) L2[i] / n) / 2;
        if (P != 0) {
            klDiv += P * log(P / Q);
        }
    }
    return klDiv;
}

__device__ double jsDivergence(const int *L1, const int *L2, int n, int maxdata) {
    // 计算JS散度
    double klDivP = klDivergence(L1, L2, n, maxdata);
    double klDivQ = klDivergence(L2, L1, n, maxdata);
    return (klDivP + klDivQ) / 2;
}


__global__ void
calculateProbabilityDistribution_Int_processKernel(const int *data, int *probabilities, int *max_list, int n) {
    unsigned int k_index = blockIdx.x;
    unsigned int d_index = threadIdx.x;

    unsigned int k = gridDim.x;
    unsigned int d = blockDim.x;
    for (int i = 0; i < n; ++i) {
        int value = data[d_index * n * k + k * i + k_index];
        ++probabilities[d_index * k * (d + 1) + k_index * (d + 1) + value];
        unsigned int max_index = d_index * k + k_index;
        max_list[max_index] = (max_list[max_index] > value) ? max_list[max_index] : value;
    }
}

__global__ void
calculateEpsilon_Int_processKernel(int *probabilities, const int *max_list, double *epsilons, int n, int k) {
    unsigned int d_index_1 = blockIdx.x;
    unsigned int d_index_2 = threadIdx.x;

    unsigned int d = blockDim.x;

    double epsilon = 0.0;
    if (d_index_2 > d_index_1) {
        for (int i = 0; i < k; ++i) {
            int *L1 = probabilities + (k * d_index_1 * (d + 1) + i * (d + 1));
            int *L2 = probabilities + (k * d_index_2 * (d + 1) + i * (d + 1));

            int maxdata =
                    max_list[k * d_index_1 + i] > max_list[k * d_index_2 + i] ? max_list[k * d_index_1 + i] : max_list[
                            k * d_index_2 + i];

            double epsilon_i = 0;

            if (jsDivergence(L1, L2, n, maxdata) > 0.0001) {
                for (int j = 0; j < maxdata; ++j) {
                    double ep = abs(log((double) *(L1 + j) / n) - log((double) *(L2 + j) / n));
                    epsilon_i = epsilon_i > ep ? epsilon_i : ep;
                }
                epsilon += epsilon_i;

            }
        }
        epsilons[d_index_1 * (2 * d - d_index_1 - 1) / 2 + (d_index_2 - d_index_1 - 1)] = epsilon;
    }
}

extern "C"
void
calculateProbabilityDistribution_Int(const vector<int> &cpu_data, vector<double> &cpu_epsilons, int d, int n,
                                     int k) {
    // 读取数据...
    // 分配和拷贝到GPU
    int *gpu_data;
    cudaMalloc(&gpu_data, cpu_data.size() * sizeof(int));
    cudaMemcpy(gpu_data, cpu_data.data(), cpu_data.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 为概率分配内存
    int *probabilities;
    cudaMalloc(&probabilities, k * (d + 1) * d * sizeof(int));
    cudaMemset(probabilities, 0, k * (d + 1) * d * sizeof(int));
    // 为max_list分配内存
    int *max_list;
    cudaMalloc(&max_list, k * d * sizeof(int));
    cudaMemset(&max_list, 0, k * d * sizeof(int));
    // 为epsilon分配内存
    double *epsilons;
    cudaMalloc(&epsilons, d * (d - 1) / 2 * sizeof(double));
    cudaMemset(&epsilons, 0.0, d * (d - 1) / 2 * sizeof(double));

    // 启动内核
    calculateProbabilityDistribution_Int_processKernel<<<k, d>>>(gpu_data, probabilities, max_list, n);
    cudaDeviceSynchronize();

    calculateEpsilon_Int_processKernel<<<d, d>>>(probabilities, max_list, epsilons, n, k);

    // 将结果拷贝回主机内存
    cudaMemcpy(cpu_epsilons.data(), epsilons, d * (d - 1) / 2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 清理GPU内存
    cudaFree(gpu_data);
    cudaFree(probabilities);
    cudaFree(max_list);
    cudaFree(epsilons);
}