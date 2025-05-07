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
float8_t f8Zero = {0, 0, 0, 0, 0, 0, 0, 0};

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
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int vert_par = 1;
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

    printf("right answer \n");
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            std::cout << rec_sum[y][x] << " ";
        }
        std::cout << "\n";
    }

    printf("\n\n vec regs one\n");
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < (7 + nx) / 8; x++) {
            for (int i = 0; i < 8; i++) {
                std::cout << rec_sum_vec[y][x][i] << " ";
            }
        }
        std::cout << "\n";
    }
    const float o = rec_sum[ny - 1][nx - 1];
    const float8_t total_sum = {o, o, o, o, o, o, o, o};
    auto start2 = std::chrono::high_resolution_clock::now();

    //#pragma omp parallel
    {
        float8_t long_rec_sum[vert_par];
        float8_t total_sse[vert_par];
        float8_t sum[vert_par];
        float8_t rec_size[vert_par];
        float8_t background_size[vert_par];
        float8_t wide_rec_sum[vert_par];
        float8_t small_rec_sum[vert_par];
        for (int vert_y = 0; vert_y < vert_par; vert_y++) {
            sum[vert_y] = f8Zero;
            long_rec_sum[vert_y] = f8Zero;
            rec_size[vert_y] = f8Zero;
            wide_rec_sum[vert_y] = f8Zero;
            small_rec_sum[vert_y] = f8Zero;
            background_size[vert_y] = f8Zero;
        }
        float8_t background_sum[vert_par];
        float8_t rec_sse;
        float8_t background_sse;
        int thread = omp_get_thread_num();
        float lowest_score = 10e+5;
        //#pragma omp for schedule(dynamic, 1)
        for (int y1 = 0; y1 < ny; y1++) {
            for (int x1 = 0; x1 < (7 + nx) / 8; x1++) {
                for (int y0 = 0; y0 <= y1; y0 += vert_par){
                    for (int x0 = 0; x0  <= x1 * 8 + 8; x0++) {
                        // printf("\n\nx0 in start: %i\n", x0);
                        float long_rec_sum_float = x0 > 0 ? rec_sum[y1][x0 - 1] : 0;                              

                        for (int vert_y = 0; vert_y < vert_par; vert_y++) {
                            if (vert_y + y0 > y1) continue;
                            sum[vert_y] = rec_sum_vec[y1][x1];


                            for (int i = 0; i < 8; i++) {
                                long_rec_sum[vert_y][i] = long_rec_sum_float;
                            }

                            if (y0 + vert_y > 0 && x0 - 1 > 0) {
                                float s = rec_sum[y0 - 1 + vert_y][x0 - 1];
                                for (int i = 0; i < 8; i++) {
                                    small_rec_sum[vert_y][i] = s;
                                }
                            } else {
                                for (int i = 0; i < 8; i++) {
                                    small_rec_sum[vert_y][i] = 0;
                                }
                            }

                            sum[vert_y] -= long_rec_sum[vert_y];
                            if (y0 + vert_y > 0) {
                                sum[vert_y] += small_rec_sum[vert_y];
                                sum[vert_y] -= rec_sum_vec[y0-1+vert_y][x1];
                            }
                        }
                        for (int vert_y = 0; vert_y < vert_par; vert_y++) {
                            if (vert_y + y0 > y1) continue;
                            for (int i= 0; i < 8; i++) {
                                rec_size[vert_y][i] = (8 * x1 + i - x0 > 0) ? (8 * x1 + i - x0 + 1) * (y1-(y0 + vert_y) + 1) : 1;
                                background_size[vert_y][i] = ny * nx - rec_size[vert_y][i];
                            }
                        }

                    
                        
                        for (int vert_y = 0; vert_y < vert_par; vert_y++) {
                            if (vert_y + y0 > y1) continue;
                        background_sum[vert_y] = total_sum - sum[vert_y];
                        rec_sse = sum[vert_y] - ((sum[vert_y] * sum[vert_y]) / (rec_size[vert_y]));
                        background_sse = background_sum[vert_y] - ((background_sum[vert_y] * background_sum[vert_y]) / background_size[vert_y]);
                        total_sse[vert_y] = rec_sse + background_sse;
                        for (int i = 0; i < 8; i++) {
                            if (x0 > x1 * 8 + i || i + x1 * 8 >= nx) continue;
                            printf("y1: %i, y0: %i , sum: %f", y1, y0 + vert_y, sum[vert_y][i]);
                            printf("x1: %i, x0: %i, total_sse: %f,  i + x1 * 8 >= nx: %d\n", x1 * 8 + i, x0, total_sse[vert_y][i], i + x1 * 8 >= nx);
                            if (total_sse[vert_y][i] < lowest_score) {
                                min_thread[thread] = total_sse[vert_y][i];                                                        
                                lowest_score = total_sse[vert_y][i];
                                res[thread].y0 = y0 + vert_y;
                                res[thread].x0 = x0;
                                res[thread].y1 = y1 + 1;
                                res[thread].x1 = x1 * 8 + i + 1;
                                res[thread].inner = sum[vert_y][i] / rec_size[vert_y][i];
                                res[thread].outer = background_sum[vert_y][i] / background_size[vert_y][i];
                            }
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
    // std::cout << result.inner[0] << " " <<  result.outer[1] << "\n";
return result;
}