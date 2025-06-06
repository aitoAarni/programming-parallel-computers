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
float rec_sum[600 * 600];
ResultD smallest_results[600 * 600];
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
            if (y0 > y1 || x0 > x1) {
                continue;
            }
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
            if (x0 == 0 && y0 == 0 && y1 == 2 && x1 == 0) {
                // printf("whole rec: %f, sum: %f\n", wholeRec, sum);
            }
            int recArea = (x1 - x0 + 1) * (y1 - y0 + 1);

            int backgroundArea = nx * ny - recArea;
            
            
            if (backgroundArea == 0) {
                backgroundArea = 1;
            }
            float backgroundSum = totalSum - sum;
            float rec_sse = sum - sum * sum / recArea;
            float background_sse = backgroundSum  - backgroundSum * backgroundSum / backgroundArea;
            printf("x0: %d, y0: %d, x1: %d, y1: %d, rec_sse: %f, background_sse: %f\n", x0, y0, x1, y1, rec_sse, background_sse);
            float sse = rec_sse + background_sse;
            if (sse < smallest.sse) {
                smallest.sse = sse;
                smallest.y0 = y0;
                smallest.x0 = x0;
                smallest.y1 = y1 + 1;
                smallest.x1 = x1 + 1;
                smallest.outer = backgroundSum / backgroundArea;
                smallest.inner = sum / recArea;
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
                sum += rec_sum[y * nx + x-1];
            }
            if (y > 0) {
                sum += rec_sum[(y-1) * nx + x];
                if (x > 0) {
                    sum -= rec_sum[(y-1) * nx + x-1];
                }
            }
            sum += data[baseIndex];
            rec_sum[y * nx + x] = sum;
        }
    }

    float* dGPU = NULL;
    ResultD* rGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nx * ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&rGPU, nx * ny * sizeof(ResultD)));
    CHECK(cudaMemcpy(dGPU, rec_sum, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 8);
    dim3 dimGrid(nx, ny);
    mykernel<<<dimGrid, dimBlock>>>(dGPU, rGPU, nx, ny);

    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(smallest_results, rGPU, nx * ny * sizeof(ResultD), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));

    int smallestCoord = 0;
    float currentSmallest = 10e5;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            if (smallest_results[y * nx + x].sse < currentSmallest) {
                currentSmallest = smallest_results[y * nx + x].sse;
                smallestCoord = y * nx + x;
            }
        }
    }
    ResultD smallest = smallest_results[smallestCoord]; 
    result.y0 = smallest.y0;
    result.x0 = smallest.x0;
    result.y1 = smallest.y1;
    result.x1 = smallest.x1;
    result.outer[0] = smallest.outer;
    result.outer[1] = smallest.outer;
    result.outer[2] = smallest.outer;
    result.inner[0] = smallest.inner;
    result.inner[1] = smallest.inner;
    result.inner[2] = smallest.inner;
    return result;
}