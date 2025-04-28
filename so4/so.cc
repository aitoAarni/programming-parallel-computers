#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>

typedef unsigned long long data_t;

static void merge(int start, int mid, int end, data_t* data, data_t* temp) {
    int leftEnd = mid - start + 1;
    int rightEnd = end - mid;

    for (int i = 0; i < leftEnd; i++) {
        temp[start + i] = data[start + i];
    }
    for (int i = 0; i < rightEnd; i++) {
        temp[mid + 1 + i] = data[mid + 1 + i];
    }

    int leftP = start;
    int rightP = mid + 1;
    int idx = start;

    while (leftP <= mid && rightP <= end) {
        if (temp[leftP] <= temp[rightP]) {
            data[idx++] = temp[leftP++];
        } else {
            data[idx++] = temp[rightP++];
        }
    }

    while (leftP <= mid) {
        data[idx++] = temp[leftP++];
    }
    while (rightP <= end) {
        data[idx++] = temp[rightP++];
    }
}

static void mergeSort(int start, int end, data_t* data, data_t* temp) {
    if (start >= end) return;

    int mid = (start + end) / 2;
    if (end - start < 10000) {
        std::sort(data + start, data + end + 1);
    } else {
        #pragma omp task shared(data, temp)
        mergeSort(start, mid, data, temp);

        #pragma omp task shared(data, temp)
        mergeSort(mid + 1, end, data, temp);

        #pragma omp taskwait
        merge(start, mid, end, data, temp);
    }
}

void psort(int n, data_t* data) {
    if (n == 0) return;

    std::vector<data_t> temp(n);

    #pragma omp parallel
    #pragma omp single
    {
        mergeSort(0, n - 1, data, temp.data());
    }
}
