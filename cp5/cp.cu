#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

using namespace std::chrono;
static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)
float d[160000000];
float transpose[160000000];
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

__global__ void mykernel(int nn, int ny, int nx, const float *transpose, float *result) {
    const int ms = 4;
    __shared__ float shared1[ms][64];
    __shared__ float shared2[ms][64];
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
    for (int k = 0; k < nx; k += ms) {
        #pragma unroll
        for (int i = 0; i < ms; i++) {
            shared1[i][tx * 8 + ty] = transpose[by + ty + 8 * tx + (i + k) * nn];
            shared2[i][ty * 8 + tx] = transpose[bx + tx + 8 * ty + (i + k) * nn];
        }
        __syncthreads();
        #pragma unroll
        for (int rotation = 0; rotation < ms && rotation + k < nx; rotation++) {
            for (int i = 0; i < 8; i++) {
                int v1Col = ty + i * 8;
                int v2Col = tx + i * 8;
                v1[i] = shared1[rotation][v1Col];
                v2[i] = shared2[rotation][v2Col];
            }
            #pragma unroll
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    vv[y][x] += v1[y] * v2[x];
                }
            }
        }
        __syncthreads();
    }
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int j = by + ty + y * 8;
            int i = bx + tx + x * 8; 
            if (i >= ny || j >= ny) break;
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

//__global__ void precompute(int nn, int ny, int nx, const float data)

void correlate(int ny, int nx, const float *data, float *result) {
    int nn = roundup(ny, 64);
    auto start1 = high_resolution_clock::now();

    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
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

    auto end1 = high_resolution_clock::now();
    auto start2 = high_resolution_clock::now();

    float* tGPU = NULL;
    CHECK(cudaMalloc((void**)&tGPU, nx * nn * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemset(rGPU, 0, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(tGPU, transpose, nx * nn * sizeof(float), cudaMemcpyHostToDevice));
    auto end2 = high_resolution_clock::now();

    dim3 dimBlock(8, 8);
    dim3 dimGrid(divup(ny, 64), divup(ny, 64));
    auto start3 = high_resolution_clock::now();

    mykernel<<<dimGrid, dimBlock>>>(nn, ny, nx, tGPU ,rGPU);
    auto end3 = high_resolution_clock::now();

    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(tGPU));
    CHECK(cudaFree(rGPU));
    auto duration1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
    auto duration2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
    auto duration3 = std::chrono::duration<double, std::milli>(end3 - start3).count();

    std::printf("first: %f  second %f  third: %f", duration1, duration2, duration3);
}
