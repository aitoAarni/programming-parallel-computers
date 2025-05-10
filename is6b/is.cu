struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

struct ResultD {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer;
    float inner;
    float sse;
};
float rec_sum[600][600];
float smallest_results[600][600];
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

__global__ void mykernel(float *rec_sum, ResultD *result, int  nx, int  ny) {
    __shared__ float minShared[8 * 32];
    int x1 = blockIdx.x;
    int y1 = blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int width = blockDim.x;
    int height = blockDim.y;
    ResultD smallest;
    smallest.sse = 10e5;
    float wholeRec = rec_sum[y1 * nx + x1]; 
    float totalSum = rec_sum[(ny - 1) * nx + nx - 1];
    for (int y0 = 0 + threadY; y0 <= y1; y0 += height) {
        for (int x0 = 0 + threadX; x0 <= x1; x0 += width) {
            float sum = wholeRec;
            if (y0 > 0) {
                sum -= rec_sum[(y0 - 1) * nx + x1];
            }
            if (x0 > 0) {
                sum -= rec_sum[y1 * nx + x0 - 1];
            }
            
            if (x0 > 0 && y0 > 0) {
                sum += rec_sum[(y0 - 1) * nx + x0 - 1];
            }
            
            int recArea = (x1 - x0 + 1) * (y1 - y0 + 1);
            int backgroundArea = nx * ny - recArea;
            
            float backgroundSum = totalSum - sum;
            float rec_sse = sum - sum * sum / recArea;
            float background_sse = backgroundSum  - backgroundSum * backgroundSum / backgroundArea;
            float sse = rec_sse + background_sse;
            if (sse < smallest.sse) {
                smallest.sse = sse;
                smallest.y0 = y0;
                smallest.x0 = x0;
                smallest.y1 = y1 + 1;
                smallest.x1 = x1 + 1;
                smallest.outer = backgroundSum;
                smallest.inner = sum;
            }
        }
    }
    
    minShared[threadX + threadY * 32] = smallest.sse;
    __syncthreads();
    float smallestSSE = 10e5;
    int smallestIndex = -1;
    for (int i = 0; i < width * height; i++) {
        if (minShared[i] < smallestSSE) {
            smallestSSE = minShared[i];
            smallestIndex = i;
        }
    }
    __syncthreads();
    if (threadY == smallestIndex / width && threadX == smallestIndex % width) {
        result[x1 + y1 * nx].sse = smallest.sse;
        result[x1 + y1 * nx].y0 = smallest.y0;
        result[x1 + y1 * nx].x0 = smallest.x0;
        result[x1 + y1 * nx].y1 = smallest.y1;
        result[x1 + y1 * nx].x1 = smallest.x1;
        result[x1 + y1 * nx].outer = smallest.outer;
        result[x1 + y1 * nx].inner = smallest.inner;
    }
} 


Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    constexpr int vert_par = 4;
    for (int y = 0; y<ny; y++) {
        for (int x = 0; x < nx; x++) {
            int baseIndex = x*3 + y*nx*3;
            float sum = 0;
            if (x > 0) {
                sum += rec_sum[y][x-1];
            }
            if (y > 0) {
                sum += rec_sum[y-1][x];
                if (x > 0) {
                    sum -= rec_sum[y-1][x-1];
                }
            }
            sum += data[baseIndex];
            rec_sum[y][x] = sum;
        }
    }

    float* dGPU = NULL;
    ResultD* rGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nx * ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&rGPU, nx * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, rec_sum, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 8);
    dim3 dimGrid(nx, ny);
    mykernel<<<dimGrid, dimBlock>>>(dGPU, rGPU, nx, ny);

    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(smallest_results, rGPU, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}
