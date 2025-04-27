#include <algorithm>
#include <iostream>

typedef unsigned long long data_t;

void merge(int start, int mid, int end, data_t *data) {
    int leftP = 0;
    int rightP = 0;
    int leftEnd = mid - start + 1;
    int rightEnd = end - mid;
    int dataLeft[leftEnd];
    int dataRight[rightEnd];
    for (int i = 0; i < leftEnd; i++) {
        dataLeft[i] = data[i + start];
    }
    for (int i = 0; i < rightEnd; i++) {
        dataRight[i] = data[i + mid + 1];
    }
    int i = start;
    for (; i <= end; i++) {
        std::cout << "index: " << i << "\n";
        if (dataLeft[leftP] < dataRight[rightP]){ 
            std::cout << "smaller was:  " << dataLeft[leftP] << "  <  " << dataRight[rightP] << "\n";
            data[i] = dataLeft[leftP];
            leftP++;
            if (leftP == leftEnd) break;
        } else {
            std::cout << "smaller was:  " << dataRight[rightP] << "  <  " << dataLeft[leftP]  << "\n";
            data[i] = dataRight[rightP];
            rightP++;
            if (rightP == rightEnd) break;
        }
        std::cout << "\n";
    }
    i++;

    for (int j = leftP; j < leftEnd; j++) {
        data[i] = dataLeft[j];
        i++;
    }
    
    for (int j = rightP; j < rightEnd; j++) {
        data[i] = dataRight[j];
        i++;
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
    std::sort(data, data + n);
}

int main() {
    data_t d[13] = {1, 4, 2, 3, 4, 5, 1, 3, 5, 6, 1, 642, 6};
    mergeSort(0, 12, d);

    for (int i = 0; i < 13; i++) {
        std::cout << d[i] << " ";
    }
    return 0;
}