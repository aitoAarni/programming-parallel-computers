#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)
float zeroNormalized[160000000];
float squareNormalized[160000000];
float transpose[160000000];
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

__global__ void mykernel(int ny, int nx, const float *data, const float *transpose, float *result) {
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (bx + tx >= ny || by + ty >= ny || by + ty > bx + tx) return;
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
            if (v1Col >= ny || v2Col >= ny) break;
            v1[i] = transpose[v1Col + k * ny];
            v2[i] = transpose[v2Col + k * ny];

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

void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<float> means(ny, 0.0);
    std::vector<float> squareSums(ny, 0.0);
    for (int y = 0; y < ny; y++) {
        float sum = 0;
        for (int x = 0; x < nx; x++) {
            sum += data[x+y*nx];
        }
        means[y] = sum / nx;
    }
    
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
           zeroNormalized[x+y*nx] = data[x+y*nx] - means[y]; 
        }
    }
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            squareSums[y] += std::pow(zeroNormalized[x+y*nx], 2);           
        }
    }
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            squareNormalized[x + y * nx] = zeroNormalized[x+y*nx] / std::sqrt(squareSums[y]);
            transpose[x * ny + y] = squareNormalized[x + y * nx];
        }
    }

    float* dGPU = NULL;
    float* tGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nx * ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&tGPU, nx * ny * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemset(rGPU, 0, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, squareNormalized, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(tGPU, transpose, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(8, 8);
    dim3 dimGrid(divup(ny, 64), divup(ny, 64));
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, tGPU ,rGPU);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(tGPU));
    CHECK(cudaFree(rGPU));

}
