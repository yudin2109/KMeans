# KMeans Paralleled
My implemention of [k-means algorithm](https://en.wikipedia.org/wiki/K-means_clustering "K-means_clustering") on C++

Places with *O(n * k * d)* complexity was designed with the support of multithreading using [OpenMP](https://www.openmp.org/) API

## Usage Example
```cpp
using Point = std::array<float, 2>;

int main() {
    std::vector<Point> X = {
        {1, 0},
        {1, 4},
        {1, 2},
        {10, 0},
        {10, 4},
        {10, 2},
    };

    KMeans<Point> kmeans(2);
    kmeans.Fit(X);
    kmeans.PrintClusterCenters();
    for (auto prediction : kmeans.Predict(X))
        std::cout << prediction << " ";
}
```

```cpp
[[ 10 2 ]
 [ 1 2 ]]
1 1 1 0 0 0
```

## Proof of correctness
For data set in [simple example](simple_example.in)
![alt text](https://github.com/yudin2109/KMeans/blob/main/simple_example_plot.png)

## Time measurements
| Data set               | 1 Thread  | 2 Threads | 3 Threads | 4 Threads | 5 Threads | 6 Threads | 7 Threads | 8 Threads |
| -----------------------|:---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:|
| [data1](data/data1.in) |  0.00015s |  0.00009s |  0.00008s |  0.00010s |  0.00009s |  0.00010s |  0.00010s |  0.00008s | 
| [data2](data/data2.in) |  0.08509s |  0.04447s |  0.03130s |  0.02488s |  0.03906s |  0.03321s |  0.02842s |  0.02495s | 
| [data3](data/data3.in) |  0.64957s |  0.33335s |  0.22821s |  0.17233s |  0.28226s |  0.24174s |  0.20276s |  0.18392s | 
| [data4](data/data4.in) | 17.79283s |  8.96646s |  6.13640s |  4.61027s |  7.07813s |  6.21119s |  5.45734s |  4.89643s | 
| [data5](data/data5.in) | 59.50199s | 34.38349s | 23.75939s | 16.71419s | 25.80533s | 20.95830s | 19.57199s | 18.72677s |

## Acceleration plots
Where `acceleration_i = Time(i threads) / Time(1 thread)`

![alt text](https://github.com/yudin2109/KMeans/blob/main/data4_stats.png)

![alt text](https://github.com/yudin2109/KMeans/blob/main/data5_stats.png)

## Documentation
```cpp
template <typename SampleType>
class KMeans{
    KMeans(int nClusters_, int nThreads_ = omp_get_max_threads(), bool isTesting = false)
}
```

#### Template parameters
 * **_SampleType : iterable of numeric type_**
    
    For example: `std::vector<double>` or `std::array<float, 2>`

#### Parameters
 * **_nClusters_: int_**
 
    Number of clusters for assignment
    
 * **_nThreads_: int_**
 
    Number of threads for paralleled methods
 
 * **_isTesting: bool_**
 
    If `true` the random state of generator is fixed

```cpp
template <typename SampleCollection>
    void Fit(const SampleCollection& data)
    \\ Chages cluster center using k_means algorithm
```

#### Template parameters
 * **_SampleCollection : iterable of SampleType_**
    
    For example: `std::vector<Sample>`

#### Parameters
 * **_data_: SampleCollection_**
 
    A set of samples for fitting
 
 ```cpp
template <typename SampleCollection>
    std::vector<int> Predict(const SampleCollection& data)
    \\ Gives the predictions for samples in data
```

#### Template parameters
 * **_SampleCollection : iterable of SampleType_**
    
    For example: `std::vector<Sample>`

#### Parameters
 * **_data_: SampleCollection_**
 
    A set of samples
#### Returns
 * A vector of predictions
