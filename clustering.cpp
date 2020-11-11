#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include "omp.h"

#include "KMeans.h"

const auto fileName = "data/data5.in";
const auto outputFileName = "predictions.out";

using Point = std::array<float, 2>;


template <typename PointCollection>
double Test(KMeans<Point>& kmeans, const PointCollection& data, int N) {
    auto start = omp_get_wtime();
    for (int i = 0; i < N; ++i) {
        kmeans.Fit(data);
    }
    auto total_time = omp_get_wtime() - start;
    return total_time / N;
}


int main() {
    int nPoints, nClusters;
    std::ifstream ifs(fileName, std::ifstream::in);
    ifs >> nPoints >> nClusters;
    std::vector<Point> pointsData(nPoints);
    for (auto& p : pointsData)
        ifs >> p[0] >> p[1];
    ifs.close();

    for (int i = 1; i <= 8; ++i) {
        KMeans<Point> kmeans(nClusters, i, true);

        auto average_time = Test(kmeans, pointsData, 1);
        std::cout << average_time << std::endl;
    }
    std::cout << std::endl;


    // auto start = omp_get_wtime();
    // kmeans.Fit(pointsData);
    // std::cout << "Fitting time: " << omp_get_wtime() - start << "s" << std::endl;
    // //kmeans.PrintClusterCenters();
    // std::cout << "Score: " << kmeans.GetScore() << std::endl;

    // auto predictions = kmeans.Predict(pointsData);
    // std::ofstream ofs(outputFileName, std::ofstream::out);
    // for (int clusterID : predictions)
    //     ofs << clusterID << std::endl;
    // ofs.close();

}