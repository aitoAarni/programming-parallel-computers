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
float8_t rec_sum_vec [600][76];
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
            rec_sum_vec[y][(x + 1) / 8][(x + 1) % 8] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "vector regs: \n";
    auto end = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x += 8) {
            for (int i = 0; i < 8; i++) {
                std::cout << rec_sum_vec[y][x / 8][i] << " ";
            }
        }
        std::cout << "\n";
    }
    
    std::cout << "\n\n  right asnwer fr fr\n";
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            std::cout << rec_sum[y][x] << " ";
        }
        std::cout << "\n";
    }

    ResultD res[22];
    double min_thread[22];
    for (int i = 0; i < 22; i ++) {
        min_thread[i] = 10e+5;
    }

    const float o = rec_sum[ny - 1][nx - 1];
    const float8_t total_sum = {o, o, o, o, o, o, o, o};
    auto start2 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {

        float8_t total_sse;
        float8_t sum;
        float8_t background_sum;
        float8_t rec_sse;
        float8_t background_sse;
        int thread = omp_get_thread_num();
        float lowest_score = 10e+5;
        #pragma omp for schedule(dynamic, 1)
        for (int y1 = 0; y1 < ny; y1++) {
            for (int x1 = 0; x1 < nx; x1++) {
                for (int y0 = 0; y0 <= y1; y0++){
                    for (int x0 = 0; x0 * 8 <= x1; x0 += 8) {
                        
                        float8_t rec_size;
                        float8_t background_size;
                        float8_t wide_sum = {0, 0, 0, 0, 0, 0, 0, 0};
                        total_sse = {0, 0, 0, 0, 0, 0, 0, 0};
                        float wide_rec_sum_float = y0 > 0 ? rec_sum[y0 - 1][x1] : 0;
                        float8_t wide_rec_sum;
                        float8_t small_rec_sum;                              
                        float8_t long_rec_sum = rec_sum_vec[y1][x0];                              
                        for (int i = 0; i < 8; i++) {
                            sum[i] = rec_sum[y1][x1];
                            wide_rec_sum[i] = wide_rec_sum_float;
                        }
                        if (y0 > 0) {
                            small_rec_sum = rec_sum_vec[y0 - 1][x0];
                        } else {
                            for (int i = 0; i < 8; i++) {
                                small_rec_sum[i] = 0;
                            }
                        }

                        sum -= wide_rec_sum;
                        sum -= long_rec_sum;
                        sum += small_rec_sum;
                        
                        for (int i= 0; i < 8; i++) {
                            rec_size[i] = (x1-x0+1 + i) * (y1-y0 + 1);
                            background_size[i] = ny * nx - rec_size[i];
                        }

                    
                    
                        background_sum = total_sum - sum;
                        rec_sse = sum - ((sum * sum) / (rec_size));
                        background_sse = background_sum - ((background_sum * background_sum) / background_size);

                        for (int i = 0; i < 8; i++) {
                            if (rec_sse[i] + background_sse[i] < lowest_score) {
                                min_thread[thread] = rec_sse[i] + background_sse[i];                                                        
                                lowest_score = rec_sse[i] + background_sse[i];
                                res[thread].y0 = y0 + i;
                                res[thread].x0 = x0 + i;
                                res[thread].y1 = y1 + 1;
                                res[thread].x1 = x1 + 1;
                                res[thread].inner = sum[i] / rec_size[i];
                                res[thread].outer = background_sum[i] / background_size[i];
                            }
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