#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

__global__ void mykernel(int nn, int ny, int nx, const float *transpose, float *result) {
    int bx = blockIdx.x * 64;
    int by = blockIdx.y * 64;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (bx + tx >= ny || by + ty >= ny || by > bx) return;
    float v1[8];
    float v2[8];
    float vv[8][8];
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            vv[y][x] = 0;
        }
    }
    for (int k = 0; k < nx; k++) {
        for (int i = 0; i < 8; i++) {
            int v1Col = by + ty + i * 8;
            int v2Col = bx + tx + i * 8;
            v1[i] = transpose[v1Col + k * nn];
            v2[i] = transpose[v2Col + k * nn];

        }
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                vv[y][x] += v1[y] * v2[x];
            }
        }
    }
    for (int y = 0; y < 8; y++) {
        int j = by + ty + y * 8;
        if (j >= ny) return;
        for (int x = 0; x < 8; x++) {
            int i = bx + tx + x * 8; 
            if (i >= ny) break;
            result[j * ny + i] = vv[y][x];
        }
    }
}

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}


__global__ void preprocess(int ny, int nx, int nn, const float* data, float* d, float* transpose) {
        int y = blockIdx.x * 256 + threadIdx.x;
        if (y >= nn) return;
        if (y >= ny) {
            for (int x = 0; x < nx; x++) {
                transpose[x * nn + y] = 0;
            }
            return;
        }
        float sum = 0;
        for (int x = 0; x < nx; x++) {
            sum += data[x+y*nx];
        }

        float mean = sum / nx;
        for (int x = 0; x < nx; x++) {
           d[x+y*nx] = data[x+y*nx] - mean; 
        }

        float squareSum = 0;
        for (int x = 0; x < nx; x++) {
            squareSum += std::pow(d[x+y*nx], 2);           
        }

            for (int x = 0; x < nx; x++) {
                d[x + y * nx] = d[x+y*nx] / std::sqrt(squareSum);
                transpose[x * nn + y] = d[x + y * nx];
        }
        
}


void correlate(int ny, int nx, const float *data, float *result) {
    float* dataGPU;
    CHECK(cudaMalloc(&dataGPU, nx * ny * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    int nn = roundup(ny, 64);
    float* dGPU;
    float* tGPU;
    CHECK(cudaMalloc(&dGPU, nx * ny * sizeof(float)));
    CHECK(cudaMalloc(&tGPU, nx * nn * sizeof(float)));

    dim3 preBlock(256);
    dim3 preGrid(roundup(ny, 256));  
    preprocess<<<preGrid, preBlock>>>(ny, nx, nn, dataGPU, dGPU, tGPU);
    CHECK(cudaGetLastError());


    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemset(rGPU, 0, ny * ny * sizeof(float)));

    dim3 dimBlock(8, 8);
    dim3 dimGrid(divup(ny, 64), divup(ny, 64));

    mykernel<<<dimGrid, dimBlock>>>(nn, ny, nx, tGPU, rGPU);

    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    

    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(tGPU));
    CHECK(cudaFree(rGPU));
    

    // std::printf("first: %f  second %f  third: %f", duration1, duration2, duration3);
}