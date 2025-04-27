#include <algorithm>
#include <vector>
#include <iostream>
typedef unsigned long long data_t;


void merge(int start, int mid, int end, data_t *data) {
    int leftP = 0;
    int rightP = 0;
    int leftEnd = mid - start + 1;
    int rightEnd = end - mid;
    std::vector<data_t> dataLeft(leftEnd);
    std::vector<data_t> dataRight(rightEnd);
    for (int i = 0; i < leftEnd; i++) {
        dataLeft[i] = data[i + start];
    }
    for (int i = 0; i < rightEnd; i++) {
        dataRight[i] = data[i + mid + 1];
    }
    int i = start;
    for (; i <= end; i++) {
        if (dataLeft[leftP] < dataRight[rightP]){ 
            data[i] = dataLeft[leftP];
            leftP++;
            if (leftP == leftEnd) break;
        } else {
            data[i] = dataRight[rightP];
            rightP++;
            if (rightP == rightEnd) break;
        }
    }

    for (int j = leftP; j < leftEnd; j++) {
        i++;
        data[i] = dataLeft[j];
    }
    
    for (int j = rightP; j < rightEnd; j++) {
        i++;
        data[i] = dataRight[j];
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