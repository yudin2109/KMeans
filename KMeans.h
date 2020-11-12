#pragma once

#include <vector>
#include <random>

#include "omp.h"

template <typename SampleType>
class KMeans{
  private:
    using ValueType = typename SampleType::value_type;
    static inline const ValueType ZERO{0};
    std::mt19937 generator;
    int nThreads;
    std::vector<std::vector<ValueType>>  clustersDistsBuffer;


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

    /** Находит индекс ближайшего кластера
     * 
     * @param [elem] Объект
    */
    int predict(const SampleType& elem) {
        clusterDistInfo nearestCluster = {0, getDistance(elem, clusterCenters.front())};
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
    * @note Распараллеленно
    * @returns true, если расстояния изменились, false иначе
    */
    template <typename SampleCollection>
    bool calcDists(const SampleCollection& data) {

        size_t clusterID;
        size_t sampleID;
        size_t dataSize = data.size();
        std::vector<ValueType> newDists(nClusters, ZERO);
        
#pragma omp parallel
        {
#pragma omp for
            for (clusterID = 0; clusterID < nClusters; ++clusterID) {
                for (int threadID = 0; threadID < nThreads; ++threadID) {
                    clustersDistsBuffer[threadID][clusterID] = ZERO;
                }
            }
    
#pragma omp for
            for (sampleID = 0; sampleID < dataSize; ++sampleID) {
                int predictedCluster = predict(data[sampleID]);
                clustersDistsBuffer[omp_get_thread_num()][predictedCluster] +=
                    getDistance(data[sampleID], clusterCenters[predictedCluster]);
            }

#pragma omp for
            for (clusterID = 0; clusterID < nClusters; ++clusterID) {
                for (int threadID = 0; threadID < nThreads; ++threadID)
                    newDists[clusterID] += clustersDistsBuffer[threadID][clusterID];
            }
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


    KMeans(int nClusters_, int nThreads_ = omp_get_max_threads(), bool isTesting = false)
        : nClusters{nClusters_}, clusterCenters(nClusters_), clusterDists(nClusters_)
    {
        generator.seed(0);
        if (!isTesting) {
            std::random_device rd;
            generator.seed(rd());
        }
        omp_set_num_threads(nThreads_);
        nThreads = nThreads_;
        clustersDistsBuffer.assign(nThreads, std::vector<ValueType>(nClusters, ZERO));
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

        using Statistics = struct{SampleType sum{}; int count = 0;};
        std::vector<std::vector<Statistics>> clustersStatsBuffer(nThreads,
                                                                std::vector<Statistics>(nClusters));
        
        bool stop_flag;
        do {
            stop_flag = true;
            size_t dataSize = data.size();
            size_t sampleID;
            size_t clusterID;
            std::vector<int> predictions(data.size());

#pragma omp parallel
            {
#pragma omp for
                for (sampleID = 0u; sampleID < dataSize; ++sampleID) {
                    predictions[sampleID] = predict(data[sampleID]); 
                }

#pragma omp for
                for (clusterID = 0; clusterID < nClusters; ++clusterID) {
                    for (int nThread = 0; nThread < omp_get_num_threads(); ++nThread) {
                        for (auto dim = 0u; dim < data[0].size(); ++ dim) {
                            clustersStatsBuffer[nThread][clusterID].sum[dim] = ZERO;
                        }
                        clustersStatsBuffer[nThread][clusterID].count = 0;
                    }
                }

#pragma omp for
                for (sampleID = 0u; sampleID < dataSize; ++sampleID) {
                    int predictedCluster = predictions[sampleID];
                    for (auto dim = 0u; dim < data[sampleID].size(); ++ dim) {
                        clustersStatsBuffer[omp_get_thread_num()][predictedCluster].sum[dim] += data[sampleID][dim];
                    }
                    clustersStatsBuffer[omp_get_thread_num()][predictedCluster].count++;
                }

#pragma omp for
                for (clusterID = 0; clusterID < nClusters; ++clusterID) {
                    for (int nThread = 1; nThread < nThreads; ++nThread) {
                        for (auto dim = 0u; dim < data[0].size(); ++ dim) {
                            clustersStatsBuffer[0][clusterID].sum[dim] +=
                                clustersStatsBuffer[nThread][clusterID].sum[dim];
                        }
                        clustersStatsBuffer[0][clusterID].count += clustersStatsBuffer[nThread][clusterID].count;
                    }
                }

#pragma omp for
                for (clusterID = 0u; clusterID < nClusters; ++clusterID) {
                    for (auto dim = 0u; dim < clustersStatsBuffer[0][clusterID].sum.size(); ++dim) {
                        if (clustersStatsBuffer[0][clusterID].count == 0)
                            clustersStatsBuffer[0][clusterID].sum[dim] = ZERO;
                        else
                            clustersStatsBuffer[0][clusterID].sum[dim]
                                /= clustersStatsBuffer[0][clusterID].count;
                    }
                    if (clustersStatsBuffer[0][clusterID].count != 0)
                        clusterCenters[clusterID] = clustersStatsBuffer[0][clusterID].sum;
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

    void PrintClusterCenters(std::ostream& output = std::cout) {
        output << "[";
        for (auto clusterID = 0u; clusterID < clusterCenters.size(); ++clusterID) {
            output << (clusterID == 0 ? "[ " : " [ ");
            for (const auto& val : clusterCenters[clusterID])
                output << val << " ";
            output << (clusterID + 1 == clusterCenters.size() ? "]" : "]\n");
        }
        output << "]\n";
    }
};
