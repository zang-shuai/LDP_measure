

/*
假设，一个函数它的输入为一个值，共有 k 种不同的输入。
输出为两种情况：
 1.输出为一个长度为 d 的向量，每个向量值均为 int。
 2.输出为一个长度为 d 的向量，每个向量值均为 double。

目前，已经将这 k 种不同的输入都运行 n 遍，得到一个 3 维数组，已经这个数组写入 bin 文件中。
生成 C++代码，进行如下操作：
 1. 读取这个bin文件，计算出这个 3 维数组的形状与值的类型。将读取到的值存入 1 维数组中。
 2. 将读取到的数据拷贝到gpu 中（cuda）。
 3. 如果输出是 int 型的，使用 cuda，统计算法在不同的输入下，输出向量每一位的不同输出的概率分布，统计结果存入 map 中，共有 d*k 个 map。
 4. 如果输出是 double 型的，不进行任何操作。
 5. cuda 代码单独写到另一个文件中，整个算法只有bin 文件的位置值是提前设定好的
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <cmath>
#include "print.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <thread>
#include <numeric>   // 用于accumulate

using namespace std;


extern "C"
void
calculateProbabilityDistribution_Int(const vector<int> &data, vector<double> &probabilityData, int d, int n, int k);


extern "C"
void
calculateProbabilityDistribution_Float(const vector<double> &cpu_data, vector<double> &cpu_epsilons, int d, int n,
                                       int k, double max_data, double min_data, double threshold, double difference);


vector<int> discretize(const vector<double> &doubleArray) {
    vector<int> intArray(doubleArray.size());
    unordered_map<double, int> doubleToIntMap;
    int nextInt = 0;
    for (int i = 0; i < doubleArray.size(); ++i) {
        double value = doubleArray[i];
        int intValue;

        // 检查是否已经映射为整数，如果是，则使用映射值；否则，为其分配新的整数值
        if (doubleToIntMap.find(value) != doubleToIntMap.end()) {
            intValue = doubleToIntMap[value];
        } else {
            intValue = nextInt;
            doubleToIntMap[value] = nextInt;
            nextInt++;
        }

        intArray[i] = intValue;
    }
    return intArray;
}

template<class T>
bool isDiscrete(vector<T> vec) {
    map<T, int> freq;
    const int threshold = 3; // 举例，如果一个数字出现超过1次，则认为是高频
//    for (double val: vec) {
//        freq[val]++;
//        if (freq[val] > threshold) {
//            return true;
//        }
//    }
    for (int i = 0; i < 1000; ++i) {
        freq[vec[i]]++;
        if (freq[vec[i]] > threshold) {
            return true;
        }
    }
    return false;
}

vector<double> getData(vector<double> vec, double range) {
    size_t n = vec.size();
    int first = n * range;
    int second = n * (1 - range);
    nth_element(vec.begin(), vec.begin() + first, vec.end());
    double p5 = vec[first];

    nth_element(vec.begin(), vec.begin() + second, vec.end());
    double p95 = vec[second];

    // 输出或处理第5%和第95%大的数
    return vector<double>{p5, p95};
}

vector<double> countEpsilon(const string &filename, char type) {
    int d, n, k;
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "无法打开文件: " << filename << endl;
        return vector<double>{0.0,};
    }

    // 读取形状和类型信息
    file.read(reinterpret_cast<char *>(&d), sizeof(d));
    file.read(reinterpret_cast<char *>(&n), sizeof(n));
    file.read(reinterpret_cast<char *>(&k), sizeof(k));
//    cout << "域大小：" << d << endl;
//    cout << "运行次数：" << n << endl;
//    cout << "每次算法输出大小：" << k << endl;

    auto start = chrono::high_resolution_clock::now();
    vector<double> epsilons(d * (d - 1) / 2);

    if (type == 'i') {
//        cout << "算法输出类型为： int" <<k<<n<<d<< endl;
        vector<int> data(k * n * d);
        file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(int));
        // 处理int类型数据
        calculateProbabilityDistribution_Int(data, epsilons, d, n, k);
    } else {
        vector<double> data(k * n * d);
        file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(double));
        bool b = isDiscrete(data);
        if (b) {
//            cout << "由于输出类型较少，算法输出类型为： int" << endl;
            const vector<int> &new_data = discretize(data);
            calculateProbabilityDistribution_Int(new_data, epsilons, d, n, k);
        } else {
            // 判断两个变量是否相同
//            double threshold = 0.001;
//            // 函数最小值
//            double difference = 0.001;

//            // 判断两个变量是否相同
//            double threshold = 0.01;
//            // 函数最小值
//            double difference = 0.005;


            double threshold = 0.01;
            // 函数最小值
            double difference = 0.007;


            const vector<double> &vector1 = getData(data, threshold);
//            cout << "算法输出类型为： double" << endl;
//            cout << "在区间: （" << vector1[0] << "， " << vector1[1] << "）内的数据将被用作核密度估计" << endl;
            calculateProbabilityDistribution_Float(data, epsilons, d, n, k, vector1[1], vector1[0], threshold,
                                                   difference);
        }
    }
    sort(epsilons.begin(), epsilons.end());
//    cout << "算法输出：" << epsilons << endl;


    chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;
//    cout << "Execution time: " << duration.count() << " seconds" << endl << endl;
    file.close();
    return epsilons;
}

void write_to_bin(const vector<vector<double>> &data, const string &filename) {
    ofstream file(filename, ios::binary);
    for (const auto &row: data) {
        file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
    }
//    cout << "输出为 bin 文件成功" << endl;
    file.close();
}

void write_to_csv(const vector<vector<double>> &data, const string &filename) {
    ofstream file(filename);
    for (const auto &row: data) {
        ostringstream ss;
        for (size_t i = 0; i < row.size(); ++i) {
            ss << row[i];
            if (i < row.size() - 1) ss << ",";
        }
        file << ss.str() << "\n";
    }
//    cout << "输出为 csv 文件成功" << endl;
    file.close();
}

void write_time(vector<double> values, string binFileName, string csvFileName) {
//    std::vector<double> values = {2.5, 1.0, 3.7, 4.2, 2.8};
//    std::string csvFileName = name + ".csv";
//    std::string binFileName = name + ".bin";

    // 写入CSV文件
    std::ofstream csvFile(csvFileName);
    if (csvFile.is_open()) {
        for (const double &value: values) {
            csvFile << value << ",";
        }
        csvFile.close();
//        std::cout << "数据已写入CSV文件: " << csvFileName << std::endl;
    } else {
//        std::cerr << "无法打开CSV文件" << std::endl;
    }

    // 写入二进制文件
    std::ofstream binFile(binFileName, std::ios::binary);
    if (binFile.is_open()) {
        binFile.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(double));
        binFile.close();
//        std::cout << "数据已写入二进制文件: " << binFileName << std::endl;
    } else {
//        std::cerr << "无法打开二进制文件" << std::endl;
    }
}

void write_to_csv_100(const vector<vector<double>> &data, const string &filename) {
    ofstream file(filename);
    for (const auto &row: data) {
        ostringstream ss;
        for (size_t i = 0; i < row.size();) {
            ss << row[i];
            if (i < row.size() - 1) ss << ",";
            i += 100;
        }
        file << ss.str() << "\n";
    }
//    cout << "输出为 csv 文件成功" << endl;
    file.close();
}

/**
 * OUE: 4.5
 * SUE: 4.3
 * RR: 0.05
 * duchi: 9.8
 * HE: 1307
 * pm: 386
 * @return
 */

int main(int argc, char *argv[]) {
    string filename = argv[1];
    char type = argv[2][0];
//    cout<< filename<<endl ;
//    cout<< type<<endl ;
    const vector<double> &eps = countEpsilon(filename, type);
    double maxValue = *std::max_element(eps.begin(), eps.end());
    double average = accumulate(eps.begin(), eps.end(), 0.0) / eps.size();
    cout << "[" << maxValue << "," << average << "]" << endl;
    return 0;






//    string filename = "/home/zangshuai/project/LDP_measure/file03.bin";
//    char type = 'f';
////    cout<< filename<<endl ;
////    cout<< type<<endl ;
//    const vector<double> &eps = countEpsilon(filename, type);
//    double maxValue = *std::max_element(eps.begin(), eps.end());
//    double average = accumulate(eps.begin(), eps.end(), 0.0) / eps.size();
//    cout << "[" << maxValue << "," << average << "]" << endl;
//    return 0;

}


//#include <stdio.h>
//
//double max_average_diff(double *L1, double *L2, int n, int m) {
//    double maxAvg = -1e9; // 初始化为一个非常小的值
//    for (int i = 0; i <= n - m; i += m) { // 确保不会越界
//        double sumDiff = 0;
//        for (int j = 0; j < m; ++j) {
//            sumDiff += L1[i + j] - L2[i + j];
//        }
//        double avgDiff = sumDiff / m;
//        printf("The maximum average difference is: %lf\n", avgDiff);
//        if (avgDiff > maxAvg) {
//            maxAvg = avgDiff; // 更新最大平均差值
//        }
//    }
//    return maxAvg;
//}
//
//int main() {
//    double L1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//    double L2[] = {2, 3, 4, 5, 6, 7, 80, 9, 99, 11};
//    int n = 10;
//    int m = 2;
//
//    double maxAvgDiff = max_average_diff(L1, L2, n, m);
//    printf("The maximum average difference is: %f\n", maxAvgDiff);
//
//    return 0;
//}

//int main() {
//
//
////    std::cout << "程序开始，将持续等待2小时..." << std::endl;
////    for (int i = 0; i < 2 * 6 * 60; ++i) {  // 6 * 60 = 360，总共360个10秒，共1小时
////        std::this_thread::sleep_for(std::chrono::seconds(10));  // 每次等待10秒
////        std::cout << "已过去 " << (i + 1) * 10 << " 秒." << std::endl;
////    }
////    std::cout << "等待完成." << std::endl;
//
//
//    auto start = chrono::high_resolution_clock::now();
//
////    vector<string> names = {"OUE", "RR", "SUE", "duchi", "OLH", "BLH", "HE", "pm"};
////    vector<char> types = {'i', 'i', 'i', 'f', 'i', 'i', 'f', 'f'};
////    vector<string> names = {"HE"};
////    vector<char> types = {'f'};
////    vector<string> epsilons = {"03", "05", "08", "10", "15", "20", "30"};
////    vector<string> names = {"pm", "HE"};
////    vector<char> types = {'f', 'f'};
////    vector<string> names = {"pm"};
////    vector<char> types = {'f'};
////    vector<string> epsilons = { "05", "08", "10", "15", "20", "30","01", "03"};
//
//
//
////    vector<string> names = {"RR", "BLH"};
////    vector<char> types = {'i', 'i'};
//
////    vector<string> names = {"HE"};
////    vector<char> types = {'f'};
//
//    vector<string> names = {"pm"};
//    vector<char> types = {'f'};
//
////    vector<string> epsilons = {"01", "03", "05", "08", "10", "15", "20", "30"};
//    vector<string> epsilons = {"05", "08", "10", "15", "20", "30", "01", "03"};
////    vector<string> epsilons = {"01"};
//
//    for (int i = 0; i < names.size(); ++i) {
//        string name = names[i];
//        char type = types[i];
//        vector<vector<double>> v1;
//        vector<vector<double>> v2;
//        vector<double> v3;
//        for (string &epsilon: epsilons) {
//            cout << "算法为： " << name << "    epsilon:" << epsilon << endl;
//            string filename = "./data/bin/" + epsilon + "/" + name + ".bin";
//
//
//            vector<double> max_s = {};
//            vector<double> avg_s = {};
//
//            auto start_count = chrono::high_resolution_clock::now();
//            for (int j = 0; j < 1; ++j) {
//                const vector<double> &eps = countEpsilon(filename, type);
//                double maxValue = *std::max_element(eps.begin(), eps.end());
//                double average = accumulate(eps.begin(), eps.end(), 0.0) / eps.size();
//                max_s.push_back(maxValue);
//                avg_s.push_back(average);
//            }
//            chrono::duration<double> duration_count = chrono::high_resolution_clock::now() - start_count;
//
//            v1.push_back(avg_s);
//            v2.push_back(max_s);
//            v3.push_back(duration_count.count());
//            cout << v1;
//            cout << v2;
//        }
//        write_to_bin(v1, "./output2/bin/avg/" + name + ".bin");
//        write_to_csv(v1, "./output2/csv/avg/" + name + ".csv");
//        write_to_bin(v2, "./output2/bin/max/" + name + ".bin");
//        write_to_csv(v2, "./output2/csv/max/" + name + ".csv");
//        write_time(v3, "./output2/bin/time/" + name + ".bin", "./output2/csv/time/" + name + ".csv");
////        write_to_csv_100(v1, "./output/csv_100/" + name + ".csv");
//    }
//
//    chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;
//    cout << "总运行时间: " << duration.count() << " seconds" << endl << endl;
//    return 0;
//}
