#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

// Better pivot selection using median-of-three
int partition(data_t *arr, int low, int high) {
    // Choose pivot using median-of-three method
    int mid = low + (high - low) / 2;
    
    // Sort low, mid, high elements
    if (arr[mid] < arr[low])
        std::swap(arr[low], arr[mid]);
    if (arr[high] < arr[low])
        std::swap(arr[low], arr[high]);
    if (arr[high] < arr[mid])
        std::swap(arr[mid], arr[high]);
    
    // Place the pivot at high-1
    std::swap(arr[mid], arr[high-1]);
    data_t pivot = arr[high-1];
    
    // Partitioning
    int i = low;
    int j = high - 1;
    
    while (true) {
        while (arr[++i] < pivot);
        while (pivot < arr[--j]);
        
        if (i >= j) break;
        std::swap(arr[i], arr[j]);
    }
    
    // Restore pivot
    std::swap(arr[i], arr[high-1]);
    return i;
}

// Optimized quicksort with appropriate threshold
void quickSort(data_t *arr, int low, int high, int depth) {
    // Use insertion sort for small arrays (much faster than std::sort for tiny arrays)
    const int INSERTION_SORT_THRESHOLD = 32;
    
    if (high - low <= INSERTION_SORT_THRESHOLD) {
        // Insertion sort for small arrays
        for (int i = low + 1; i <= high; i++) {
            data_t key = arr[i];
            int j = i - 1;
            while (j >= low && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
        return;
    }
    
    // Perform partitioning
    int pivotIdx = partition(arr, low, high);
    
    // Use parallel tasks only at higher tree levels to limit task creation overhead
    // Higher depth means closer to the root of the recursion tree
    const int PARALLEL_THRESHOLD = 8; // Tune this based on your core count
    
    if (depth > 0) {
        #pragma omp task default(none) firstprivate(arr, low, pivotIdx, depth)
        quickSort(arr, low, pivotIdx - 1, depth-1);
        
        #pragma omp task default(none) firstprivate(arr, pivotIdx, high, depth)
        quickSort(arr, pivotIdx + 1, high, depth-1);
    } else {
        // Sequential execution for lower levels
        quickSort(arr, low, pivotIdx - 1, 0);
        quickSort(arr, pivotIdx + 1, high, 0);
    }
}

void psort(int n, data_t *data) {
    // Fall back to std::sort for small arrays
    if (n <= 10000) {
        std::sort(data, data + n);
        return;
    }
    
    // Use about log2(num_cores) + 2 levels of parallelism
    int num_threads = omp_get_max_threads();
    int depth = 6;
    
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            quickSort(data, 0, n - 1, depth);
        }
    }
}