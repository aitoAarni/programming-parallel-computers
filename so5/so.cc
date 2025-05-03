#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

int partition(data_t *arr, int low, int high) {
    int mid = low + (high - low) / 2;
    
    if (arr[mid] < arr[low])
        std::swap(arr[low], arr[mid]);
    if (arr[high] < arr[low])
        std::swap(arr[low], arr[high]);
    if (arr[high] < arr[mid])
        std::swap(arr[mid], arr[high]);
    
    std::swap(arr[mid], arr[high-1]);
    data_t pivot = arr[high-1];
    
    int i = low;
    int j = high - 1;
    
    while (true) {
        while (arr[++i] < pivot);
        while (pivot < arr[--j]);
        
        if (i >= j) break;
        std::swap(arr[i], arr[j]);
    }
    

    std::swap(arr[i], arr[high-1]);
    return i;
}

void quickSort(data_t *arr, int low, int high, int depth) {
    //const int threshold = 32;
    
    //if (high - low <= threshold) {
    //    for (int i = low + 1; i <= high; i++) {
    //        data_t key = arr[i];
    //        int j = i - 1;
    //        while (j >= low && arr[j] > key) {
    //            arr[j + 1] = arr[j];
    //            j--;
    //        }
    //        arr[j + 1] = key;
    //    }
    //    return;
    //}
    
    int pivotIdx = partition(arr, low, high);
    
    
    if (depth > 0) {
        #pragma omp task
        quickSort(arr, low, pivotIdx - 1, depth-1);
        #pragma omp task
        quickSort(arr, pivotIdx + 1, high, depth-1);
    } else {
        quickSort(arr, low, pivotIdx - 1, 0);
        quickSort(arr, pivotIdx + 1, high, 0);
    }
}

void psort(int n, data_t *data) {
    int depth = 6;
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            quickSort(data, 0, n - 1, depth);
        }
    }
}