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
For data set in [data4](data/data4.in)
![alt text](https://github.com/yudin2109/KMeans/blob/main/example_plot.png)

## Time measurements
| Data set               |  1 Thread   |  2 Threads  |  3 Threads  |  4 Threads  |  5 Threads  |  6 Threads  |  7 Threads  |  8 Threads  |
| -----------------------|:-----------:| -----------:| -----------:| -----------:| -----------:| -----------:| -----------:| -----------:|
| [data1](data/data1.in) |  0.0001549s |  0.0000936s |  0.0000804s |  0.0001055s |  0.0000950s |  0.0001088s |  0.0001092s |  0.0000838s | 
| [data2](data/data2.in) |  0.0850907s |  0.0444714s |  0.0313023s |  0.0248829s |  0.0390621s |  0.0332199s |  0.0284214s |  0.0249516s | 
| [data3](data/data3.in) |  0.6495762s |  0.3333516s |  0.2282126s |  0.1723314s |  0.2822682s |  0.2417411s |  0.2027649s |  0.1839266s | 
| [data4](data/data4.in) | 17.7928305s |  8.9664694s |  6.1364073s |  4.6102756s |  7.0781330s |  6.2111925s |  5.4573416s |  4.8964306s | 
| [data5](data/data5.in) | 59.5019928s | 34.3834907s | 23.7593926s | 16.7141950s | 25.8053340s | 20.9583034s | 19.5719942s | 18.7267782s | 

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
