#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include <random>
#include <cmath>
#include <limits>

#include "omp.h"

template <typename SampleType>
class KMeans{
  private:
    using ValueType = typename SampleType::value_type;
    static inline const ValueType ZERO{0};
    std::mt19937 generator;


    /** Находит граничные значения данных
    * для более разумной последующей генерации центроидов
    */
    template <typename SampleCollection>
    std::pair<SampleType, SampleType> getLimits(const SampleCollection& data) {
        SampleType lowerLimits = data.front();
        SampleType upperLimits = lowerLimits;
        for (auto& p : data) {
            for (auto dim = 0u; dim < p.size(); ++dim) {
                lowerLimits[dim] = std::min(lowerLimits[dim], p[dim]);
                upperLimits[dim] = std::max(upperLimits[dim], p[dim]);
            }
        }
        return std::make_pair(lowerLimits, upperLimits);
    }


    // clusterDistInfo - пара{кластер, расстояние до него}
    using clusterDistInfo = std::pair<int, ValueType>;
#define defaultDistInfo clusterDistInfo{-1, std::numeric_limits<ValueType>::max()}

    clusterDistInfo distInfoMin(const clusterDistInfo& a, const clusterDistInfo& b){
        return a.second < b.second ? a : b;
    }


    /** Находит индекс ближайшего кластера
     * 
     * @param [elem] Объект
     * @note Распараллелено
    */
    int predict(const SampleType& elem) {
        clusterDistInfo nearestCluster = {0, getDistance(elem, clusterCenters.front())};
        clusterDistInfo defaultValue = nearestCluster;

#pragma omp declare reduction \
    (minDistInfo:clusterDistInfo:omp_out=distInfoMin(omp_out, omp_in)) \
    initializer(omp_priv = defaultDistInfo)

#pragma omp parallel for reduction(minDistInfo:nearestCluster)
        for (auto clusterID = 0u; clusterID < clusterCenters.size(); ++clusterID) {
            ValueType dist = getDistance(elem, clusterCenters[clusterID]);
            if (dist < nearestCluster.second) {
                nearestCluster.first = clusterID;
                nearestCluster.second = dist;
            }
        }
        return nearestCluster.first;
    }


    /** Считает суммарное расстояние от всех объектов до центроидов их кластеров 
    */
    ValueType metrics() {
        ValueType distSum{};
        for (auto& dist : clusterDists)
            distSum += dist;
        return distSum;
    }


    /**   Генерирует центроиды случайным образом
    *
    * @param [lowerLimits,upperLimits] границы для генерации 
    */
    void initCenters(SampleType lowerLimits, SampleType upperLimits) {
        for (auto dim = 0u; dim < lowerLimits.size(); ++dim) {
            std::uniform_real_distribution<ValueType> urd(lowerLimits[dim], upperLimits[dim]);
            for (auto& center : clusterCenters) {
                center[dim] = urd(generator);
            }
        }
    }


    /** Пересчитывает расстояния от объектов до центроидов их кластеров
    * 
    * @param [data] набор объектов
    * @returns true, если расстояния изменились, false иначе
    */
    template <typename SampleCollection>
    bool calcDists(const SampleCollection& data) {
        std::vector<ValueType> newDists(nClusters, ZERO);
        for (auto& p : data) {
            int predictedCluster = predict(p);
            newDists[predictedCluster] += getDistance(p, clusterCenters[predictedCluster]);
        }

        bool has_changed = newDists != clusterDists;
        clusterDists = newDists;
        return has_changed;
    }


    /** Находит квадрат евклидовой нормы 
     */
    ValueType getDistance(const SampleType& a, const SampleType& b) {
        ValueType dist{};
        for (auto dim = 0u; dim < a.size(); ++dim)
            dist += (a[dim] - b[dim]) * (a[dim] - b[dim]);
        return dist;
    }


  public:
    int nClusters;
    std::vector<SampleType> clusterCenters;
    std::vector<ValueType> clusterDists;


    KMeans(int nClusters_, int nThreads = omp_get_max_threads(), bool isTesting = false)
        : nClusters{nClusters_}, clusterCenters(nClusters_), clusterDists(nClusters_)
    {
        generator.seed(0);
        if (!isTesting) {
            std::random_device rd;
            generator.seed(rd());
        }
        omp_set_num_threads(nThreads);
    }


    /** С помощью алгоритма k-means находит оптимальные центроиды для кластеров
     * 
     * @note Распараллелено
     * @param [data] Набор объектов
    */
    template <typename SampleCollection>
    void Fit(const SampleCollection& data) {
        auto [lowerLimits, upperLimits] = getLimits(data);
        initCenters(lowerLimits, upperLimits);
        calcDists(data);
        
        bool stop_flag;
        do {
            stop_flag = true;

            using Statistics = struct{SampleType sum{}; int count = 0;};
            std::vector<Statistics> clustersStats(nClusters);

            auto data_size = data.size();
            auto clustersStatsSize = clustersStats.size();
            auto clusterCentersSize = clusterCenters.size();
            
            for (auto recordID = 0u; recordID < data_size; ++recordID) {
                int predictedCluster = predict(data[recordID]);
                for (auto dim = 0u; dim < data[recordID].size(); ++ dim) {
                    clustersStats[predictedCluster].sum[dim] += data[recordID][dim];
                }
                clustersStats[predictedCluster].count++;
            }

#pragma omp parallel
            {
#pragma omp for
                for (auto statsID = 0u; statsID < clustersStatsSize; ++statsID) {
                    for (auto dim = 0u; dim < clustersStats[statsID].sum.size(); ++dim) {
                        if (clustersStats[statsID].count == 0)
                            clustersStats[statsID].sum[dim] = ZERO;
                        else
                            clustersStats[statsID].sum[dim] /= clustersStats[statsID].count;
                    }
                }

#pragma omp for
                for (auto clusterID = 0u; clusterID < clusterCentersSize; ++clusterID) {
                    if (clustersStats[clusterID].count != 0)
                        clusterCenters[clusterID] = clustersStats[clusterID].sum;
                }
            }
            
            stop_flag = !calcDists(data);
        } while (!stop_flag);
    }


    /** Оценка качества кластеризации
     * 
     * @returns Минус сумма квадратов расстояний от объектов до центроидов их кластеров
    */
    ValueType GetScore() {
        return -metrics();
    }

    /** Определяет, к какому кластеру принадлежат объекты
     * 
     * @param [data] Набор объектов
     * @returns Вектор предсказаний
    */
    template <typename TCollection>
    std::vector<int> Predict(const TCollection& data) {
        std::vector<int> results;
        for (const auto& p : data) {
            results.push_back(predict(p));
        }
        return results;
    }

    void PrintClusterCenters() {
        std::cout << "[";
        for (auto clusterID = 0u; clusterID < clusterCenters.size(); ++clusterID) {
            std::cout << (clusterID == 0 ? "[ " : " [ ");
            for (const auto& val : clusterCenters[clusterID])
                std::cout << val << " ";
            std::cout << (clusterID + 1 == clusterCenters.size() ? "]" : "]\n");
        }
        std::cout << "]\n";
    }
};

#endif