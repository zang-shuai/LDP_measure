#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
//#include <cfloat>
//#include <iostream>
//#include "print.h"

using namespace std;
#define MAX_LIST 1000


__device__ void swap(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int partition(double arr[], int low, int high) {
    double pivot = arr[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

__device__ double quickselect(double arr[], int low, int high, int k) {
    if (low == high) {
        return arr[low];
    }

    int pivotIndex = partition(arr, low, high);

    if (k == pivotIndex) {
        return arr[k];
    } else if (k < pivotIndex) {
        return quickselect(arr, low, pivotIndex - 1, k);
    } else {
        return quickselect(arr, pivotIndex + 1, high, k);
    }
}

// 高斯核函数
__device__ double gaussian_kernel(double x) {
    return (1 / sqrt(2 * M_PI)) * exp(-0.5 * x * x);
}

// 计算方差
__device__ double variance(const double *data, size_t size) {
    double mean = 0.0;
    double var = 0.0;
    size_t i;

    for (i = 0; i < size; ++i) {
        mean += data[i];
    }
    mean /= size;

    for (i = 0; i < size; ++i) {
        var += (data[i] - mean) * (data[i] - mean);
    }
    var /= size;
    return var;
}

// 统计核函数
__global__ void
calculateProbabilityDistribution_Float_processKernel(const double *data, double *densities,
                                                     int n,
                                                     double min_val, double max_val) {
    unsigned int k_index = blockIdx.x;
    unsigned int d_index = threadIdx.x;

    unsigned int k = gridDim.x;
    const double bandwidth = 1.06 * sqrt(variance(data, n)) * pow(n, -0.2);
    for (unsigned int i = 0; i < MAX_LIST; ++i) {
        double x = min_val + i * (max_val - min_val) / MAX_LIST;
        double density = 0.0;
        for (int j = 0; j < n; ++j) {
            density += gaussian_kernel((x - data[d_index * n * k + k * j + k_index]) / bandwidth);
        }
        density /= ((double) n * bandwidth);
        densities[d_index * k * MAX_LIST + k_index * MAX_LIST + i] = density;
    }
}

__device__ double max_abs_average_diff(const double *L1, const double *L2, int n, int m) {
    double maxAvg = 0; // 初始化为一个非常小的值
//    double maxAvgValue = 0; // 用于存储绝对值最大的平均差值
    for (int i = 0; i <= n - m; i += m) { // 确保不会越界
        double sumDiff = 0;
        for (int j = 0; j < m; ++j) {
            sumDiff += L1[i + j] - L2[i + j];
        }
        double avgDiff = sumDiff / m;
        if (fabs(avgDiff) > maxAvg) {
            maxAvg = fabs(avgDiff); // 更新最大绝对值
//            maxAvgValue = avgDiff; // 保存实际的平均差值
        }
    }
    return maxAvg;
}

__device__ double average_abs_diff(const double *L1, const double *L2, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += fabs(L1[i] - L2[i]); // 计算差的绝对值并累加
    }
    return sum / n; // 计算平均值
}

// 计算 epsilon 核函数
__global__ void
calculateEpsilon_Float_processKernel(double *densities, double *epsilons, int k, double max_data, double min_data,
                                     double threshold, double difference) {
    unsigned int d_index_1 = blockIdx.x;
    unsigned int d_index_2 = threadIdx.x;
    unsigned int d = blockDim.x;
    double epsilon = 0.0;
    if (d_index_2 > d_index_1) {
        int xz=0;
        for (int i = 0; i < k; ++i) {
            double *L1 = densities + (k * d_index_1 * MAX_LIST + i * MAX_LIST);
            double *L2 = densities + (k * d_index_2 * MAX_LIST + i * MAX_LIST);
            double diff = max_abs_average_diff(L1, L2, MAX_LIST, 100);
//            printf("%f,%f\n",average_abs_diff(L1,L2,MAX_LIST),diff);
//            for (int j = 0; j < MAX_LIST; ++j) {
//                printf("%f,",L1[j]);
//            }
//            printf("\n");
//            printf("\n");
//            printf("\n");
//            printf("\n");
//            for (int j = 0; j < MAX_LIST; ++j) {
//                printf("%f,",L2[j]);
//            }
//            printf("\n");
//            printf("\n");
//            printf("\n");
//            printf("\n");
            if (diff > difference) {
//                xz++;
//                printf("diff:  %lf\n",fabs(diff));

                double epsilon_i[MAX_LIST];
                for (int j = 0; j < MAX_LIST; ++j) {
                    if (*(L1 + j) > threshold && *(L2 + j) > threshold) {
                        epsilon_i[j] = abs(log((double) *(L1 + j) / (double) *(L2 + j)));
                    }
                }
                double ep = quickselect(epsilon_i, 0, MAX_LIST - 1, MAX_LIST * 80 / 100 - 1);

                epsilon += ep;
            };
//            printf("epsilon:  %lf %d\n",epsilon,xz);

//            printf("%d\n",xz);

        }
        epsilons[d_index_1 * (2 * d - d_index_1 - 1) / 2 + (d_index_2 - d_index_1 - 1)] = epsilon;
    }
}

// C++ 核函数
extern "C"
void
calculateProbabilityDistribution_Float(const vector<double> &cpu_data, vector<double> &cpu_epsilons, int d, int n,
                                       int k,
                                       double max_data, double min_data, double threshold, double difference) {
    // 读取数据...
    // 分配和拷贝到GPU
    double *gpu_data;
    cudaMalloc(&gpu_data, cpu_data.size() * sizeof(double));
    cudaMemcpy(gpu_data, cpu_data.data(), cpu_data.size() * sizeof(double), cudaMemcpyHostToDevice);

    // 为概率分配内存
    double *densities;
    cudaMalloc(&densities, k * MAX_LIST * d * sizeof(double));
    cudaMemset(densities, 0, k * MAX_LIST * d * sizeof(double));

    // 为epsilon分配内存
    double *epsilons;
    cudaMalloc(&epsilons, d * (d - 1) / 2 * sizeof(double));
    cudaMemset(&epsilons, 0.0, d * (d - 1) / 2 * sizeof(double));

    // 启动内核
    calculateProbabilityDistribution_Float_processKernel<<<k, d>>>(gpu_data, densities, n, min_data, max_data);
    cudaDeviceSynchronize();

//    vector<int> v(d * k);
//    cudaMemcpy(v.data(), max_list, d * k * sizeof(int), cudaMemcpyDeviceToHost);
    vector<double> p(d * (d + 1) * k);
    cudaMemcpy(p.data(), densities, d * (d + 1) * k * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "max_list" << v << v.size() << endl;
//    cout << " ========= " << endl;
//    cout << "probabilities" << p << p.size() << endl;
//    cout << " ========= " << endl;



    calculateEpsilon_Float_processKernel<<<d, d>>>(densities, epsilons, k, max_data, min_data, threshold, difference);

    // 将结果拷贝回主机内存
    cudaMemcpy(cpu_epsilons.data(), epsilons, d * (d - 1) / 2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    cout << "max_list" << v << v.size() << endl;
    // 清理GPU内存
    cudaFree(gpu_data);
    cudaFree(densities);
    cudaFree(epsilons);
}