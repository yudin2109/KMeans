#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <vector>
#include <cmath>
#include "omp.h"

#include "KMeans.h"

using Point = std::array<float, 2>;

template <typename Numerics>
class Statistics {
  private:
    std::vector<Numerics> data;

  public:
    Statistics() : data(0)
    { }

    void Account(Numerics num) {
        data.push_back(num);
    }

    Numerics GetMean() const {
        Numerics sum{};
        for (auto num : data)
            sum += num;
        return data.empty() ? Numerics{} : sum / data.size();
    }

    Numerics GetSD() const {
        Numerics mean = GetMean();
        Numerics sqrSum{};
        for (auto num : data)
            sqrSum += (num - mean) * (num - mean);
        return data.empty() ? Numerics{} : sqrt(sqrSum / data.size());
    }
};

template <typename PointCollection>
std::pair<double, double> GetTimeStatistics(KMeans<Point>& kmeans, const PointCollection& data, int Nchecks) {
    Statistics<double> timeStatistics;
    for (int i = 0; i < Nchecks; ++i) {
        double timeStamp = omp_get_wtime();
        kmeans.Fit(data);
        timeStatistics.Account(omp_get_wtime() - timeStamp);
    }
    return {timeStatistics.GetMean(), timeStatistics.GetSD()};
}

const auto fileName = "data/data1.in";

int main() {
    /// Чтение данных из файла
    int nPoints, nClusters;
    std::ifstream ifs(fileName, std::ifstream::in);
    ifs >> nPoints >> nClusters;
    std::vector<Point> pointsData(nPoints);
    for (auto& p : pointsData)
        ifs >> p[0] >> p[1];
    ifs.close();

    /// Тестирование на различных количествах потоков
    /// На 4-5 тестах будет работать долго, параметр nChecks можно уменьшить до 1
    int nChecks = 5;
    for (int nThreads = 1; nThreads <= 8; ++nThreads) {
        KMeans<Point> kmeans(nClusters, nThreads, true);
        auto average_time = GetTimeStatistics(kmeans, pointsData, nChecks);
        std::cout << "Time for " << nThreads << \
            " threads: mean " << std::fixed << std::setprecision(6) << average_time.first << \
            " ; std = " << average_time.second << std::endl;
    }
    std::cout << std::endl;

    /// Вывод результата
    KMeans<Point> kmeans(nClusters);
    kmeans.Fit(pointsData);
    kmeans.PrintClusterCenters();
    std::cout << "Score: " << kmeans.GetScore() << std::endl;

    /// Предсказанные номера кластеров выводятся в отдельный файл
    auto outputFileName = "predictions.out";
    auto predictions = kmeans.Predict(pointsData);
    std::ofstream ofs(outputFileName, std::ofstream::out);
    for (int clusterID : predictions)
        ofs << clusterID << std::endl;
    ofs.close();
}