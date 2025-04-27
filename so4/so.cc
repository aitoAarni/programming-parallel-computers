#include <algorithm>
#include <iostream>
typedef unsigned long long data_t;

data_t tempData[100000000];

void merge(int start, int mid, int end, data_t *data) {
    int leftP = 0;
    int rightP = 0;
    int leftEnd = mid - start + 1;
    int rightEnd = end - mid;
    for (int i = 0; i < leftEnd; i++) {
        tempData[i + start] = data[i + start];
    }
    for (int i = 0; i < rightEnd; i++) {
        tempData[i + mid + 1] = data[i + mid + 1];
    }
    int i = start;
    for (; i <= end; i++) {
        if (tempData[leftP + start] < tempData[rightP + mid + 1]){ 
            data[i] = tempData[leftP + start];
            leftP++;
            if (leftP == leftEnd) break;
        } else {
            data[i] = tempData[rightP + mid + 1];
            rightP++;
            if (rightP == rightEnd) break;
        }
    }

    for (int j = leftP; j < leftEnd; j++) {
        i++;
        data[i] = tempData[j + start];
    }
    
    for (int j = rightP; j < rightEnd; j++) {
        i++;
        data[i] = tempData[j + mid + 1];
    }
    
}
void mergeSort(int start, int end, data_t *data) {
    if (start == end) return;

    int mid =(end + start) / 2;

    mergeSort(start, mid, data);
    mergeSort(mid + 1, end, data);
    merge(start, mid, end, data);
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    // for (int i = 0; i < n; i++) {
    //     std::cout << data[i] << " ";
    // }
    if (n == 0) return;
    mergeSort(0, n - 1, data);

}