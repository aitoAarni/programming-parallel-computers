#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

typedef float float8_t __attribute__ ((vector_size ( 8 * sizeof(float))));

struct ResultD {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer;
    float inner;
};


float rec_sum[600][600];
float8_t rec_sum_vec [600][75];
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    int vecX = (nx + 7)/ 8;
    constexpr int columnBlock = 1;
    constexpr int rowBlock = 1;
    int newX = (nx + columnBlock - 1) / columnBlock;
    auto start = std::chrono::high_resolution_clock::now();

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
            rec_sum_vec[y][x / 8][x % 8] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    

    ResultD res[22];
    double min_thread[22];
    for (int i = 0; i < 22; i ++) {
        min_thread[i] = 10e+5;
    }
    const float total_sum = rec_sum[ny - 1][nx - 1];
    const float total_square_sum = rec_sum[ny - 1][nx - 1];
    auto start2 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {

        float total_sse;
        float sum;
        float background_sum;
        float rec_sse;
        float background_sse;
        int thread = omp_get_thread_num();
        float lowest_score = 10e+5;
        #pragma omp for schedule(dynamic, 1)
        for (int y1 = 0; y1 < ny; y1++) {
            for (int x1 = 0; x1 < nx; x1++) {
            for (int y0 = 0; y0 <= y1; y0++){
                    for (int x0 = 0; x0 <= x1; x0++) {
                        
                        float rec_size;
                        float background_size;

                                total_sse = 0;
                                sum = rec_sum[y1][x1];
                                
                                
                                if (y0 > 0) {
                                sum -= rec_sum[y0-1][x1];
                            }
                            if (x0 > 0) {
                                sum -= rec_sum[y1][x0-1];
                            }
                            if (y0>0 && x0>0) {
                                sum += rec_sum[y0-1][x0-1];
                            }
                       
                            rec_size = (x1-x0+1) * (y1-y0 + 1);
                            background_size = ny * nx - rec_size;

                        
                        
                        background_sum = total_sum - sum;
                        rec_sse = sum - ((sum * sum) / (rec_size));
                        background_sse = background_sum - ((background_sum * background_sum) / background_size);
                        for (int z = 0; z < 3; z++) {
                            total_sse += rec_sse;
                            total_sse += background_sse;
                        }
                        if (total_sse < lowest_score) {
                            min_thread[thread] = total_sse;                                                        
                            lowest_score = total_sse;
                            res[thread].y0 = y0;
                            res[thread].x0 = x0;
                            res[thread].y1 = y1 + 1;
                            res[thread].x1 = x1 + 1;
                            res[thread].inner = sum / rec_size;
                            res[thread].outer = background_sum / background_size;
                        }
                    }
             }       
                }
            }
            
        }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::duration<double>(end - start);
    auto t2 =  std::chrono::duration<double>(end2 - start2);
    // printf("t1: %f   t2: %f", t1.count(), t2.count());
    double minimum = 10e+50;
    for (int i = 0; i < 22; i++) {
        if (min_thread[i] < minimum) {
            minimum = min_thread[i];
            result.y0 = res[i].y0;
            result.x0 = res[i].x0;
            result.y1 = res[i].y1;
            result.x1 = res[i].x1;
            result.inner[0] = res[i].inner;
            result.inner[1] = res[i].inner;
            result.inner[2] = res[i].inner;
            result.outer[0] = res[i].outer;
            result.outer[1] = res[i].outer;
            result.outer[2] = res[i].outer;
        }
    }
    return result;
}