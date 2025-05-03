#include <algorithm>

typedef unsigned long long data_t;
#include <bits/stdc++.h>
using namespace std;

int partition(data_t *arr, int low, int high) {
  
    data_t pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    
    swap(arr[i + 1], arr[high]);  
    return i + 1;
}

void quickSort(data_t *arr, int low, int high) {
  
    if (low < high) {
      
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
void psort(int n, data_t *data) {
    quickSort(data, 0, n - 1);
}
