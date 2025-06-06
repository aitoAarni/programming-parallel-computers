#include <iostream>
#include <iomanip>
#include <omp.h>
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double)), aligned(32)));
double4_t d4zero = {0, 0, 0, 0};
struct ResultD {
    int y0;
    int x0;
    int y1;
    int x1;
    double4_t outer;
    double4_t inner;
};

struct ThreadResult {
    double error;
    int y0, x0, y1, x1;
    double4_t inner, outer;
};

alignas(64) double4_t rec_sum[400][400];
alignas(64) double4_t sum_square[400][400];
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    constexpr int columnBlock = 3;
    constexpr int rowBlock = 1;

    for (int y = 0; y<ny; y++) {
        for (int x = 0; x < nx; x++) {
            int baseIndex = x*3 + y*nx*3;
            double4_t sum = {0, 0, 0, 0};
            double4_t square_sum = {0, 0, 0, 0};
            if (x > 0) {
                sum += rec_sum[y][x-1];
                square_sum += sum_square[y][x-1];
            }
            if (y > 0) {
                sum += rec_sum[y-1][x];
                square_sum += sum_square[y-1][x];
                if (x > 0) {
                    sum -= rec_sum[y-1][x-1];
                    square_sum -= sum_square[y-1][x-1];
                }
            }
            for (int i = 0; i<3; i++) {
                sum[i] += data[baseIndex + i];
                square_sum[i] +=  data[baseIndex + i] * data[baseIndex + i];
            }
            rec_sum[y][x] = sum;
            sum_square[y][x] = square_sum;
        }
    }

    ThreadResult res[22];
    for (int i = 0; i < 22; i ++) {
        res[i].error = 10e+5;
    }
    const double4_t total_sum = rec_sum[ny - 1][nx - 1];
    const double4_t total_square_sum = sum_square[ny - 1][nx - 1];

    #pragma omp parallel
    {
        const bool big = ny * nx > 140000;

        double total_sse[rowBlock][columnBlock];
        double4_t sum[rowBlock][columnBlock];
        double4_t square_sum[rowBlock][columnBlock];
        double4_t background_sum[rowBlock][columnBlock];
        double4_t background_square_sum[rowBlock][columnBlock];
        double4_t rec_sse[rowBlock][columnBlock];
        double4_t background_sse[rowBlock][columnBlock];
        for (int i = 0; i < rowBlock; ++i) {
    for (int j = 0; j < columnBlock; ++j) {
        total_sse[i][j] = 0.0;
        sum[i][j] = d4zero;
        square_sum[i][j] = d4zero;
        background_sum[i][j] = d4zero;
        background_square_sum[i][j] = d4zero;
        rec_sse[i][j] = d4zero;
        background_sse[i][j] = d4zero;
    }
}
        int thread = omp_get_thread_num();
        double lowest_score = 10e+5;
        #pragma omp for schedule(dynamic, 2)
        for (int height = 0; height < ny; height++) {
            for (int width = 0; width < nx; width++) {
                    double tmp =  (height + 1) * (width + 1);
                    const double4_t rec_size = {tmp, tmp, tmp, tmp};
                    tmp = ny * nx - tmp;
                    const double4_t background_size = {tmp, tmp, tmp, tmp};
                    if (big && (tmp < 401 || (height + 1) * (width + 1) < 401)) continue;
                    for (int y0 = 0; y0 < ny - height; y0 += rowBlock){
                        for (int x0 = 0 ; x0 < nx - width; x0 += columnBlock) {
                            int y1 = y0 + height;
                            int x1 = x0 + width;
                            for (int i = 0; i < rowBlock; i++) {
                                for (int j = 0; j < columnBlock; j++) {
                                
                                if (y1 + i >= ny || x1 + j >= nx) break;
                                total_sse[i][j] = 0;
                                sum[i][j] = rec_sum[y1 + i][x1 + j];
                                square_sum[i][j] = sum_square[y1 + i][x1 + j];
                                
                                
                                if (y0 + i > 0) {
                                sum[i][j] -= rec_sum[y0-1 + i][x1 + j];
                                square_sum[i][j] -= sum_square[y0-1 + i][x1 + j];   
                            }
                            if (x0 + j > 0) {
                                sum[i][j] -= rec_sum[y1 + i][x0-1 + j];
                                square_sum[i][j] -= sum_square[y1 + i][x0-1 + j];   
                            }
                            if (y0 + i>0 && x0 + j>0) {
                                sum[i][j] += rec_sum[y0-1 + i][x0-1 + j];
                                square_sum[i][j] += sum_square[y0-1 + i][x0-1 + j];   
                            }
                       
                        }
                    }
                    for (int i = 0; i < rowBlock; i++) {
                        for (int j = 0; j < columnBlock; j++) {
                        if (y1 + i >= ny || x1 + j >= nx) break;
                        
                        background_sum[i][j] = total_sum - sum[i][j];
                        background_square_sum[i][j] = total_square_sum - square_sum[i][j];
                        rec_sse[i][j] = square_sum[i][j] - ((sum[i][j] * sum[i][j]) / rec_size);
                        background_sse[i][j] = background_square_sum[i][j] - ((background_sum[i][j] * background_sum[i][j]) / background_size);
                        for (int z = 0; z < 3; z++) {
                            total_sse[i][j] += rec_sse[i][j][z] + background_sse[i][j][z];
                        }
                        if (total_sse[i][j] < lowest_score) {
                            lowest_score = total_sse[i][j];
                            res[thread].error = total_sse[i][j];
                            res[thread].y0 = y0 + i;
                            res[thread].x0 = x0 + j;
                            res[thread].y1 = y1 + 1 + i;
                            res[thread].x1 = x1 + 1 + j;
                            res[thread].inner = sum[i][j] / rec_size;
                            res[thread].outer = background_sum[i][j] / background_size;
                        }
                    }
             }       
                }
            }
            
        }
        }
    }
    double minimum = 10e+5;
    for (int i = 0; i < 22; i++) {
        if (res[i].error < minimum) {
            minimum = res[i].error;
            result.y0 = res[i].y0;
            result.x0 = res[i].x0;
            result.y1 = res[i].y1;
            result.x1 = res[i].x1;
            result.inner[0] = res[i].inner[0];
            result.inner[1] = res[i].inner[1];
            result.inner[2] = res[i].inner[2];
            result.outer[0] = res[i].outer[0];
            result.outer[1] = res[i].outer[1];
            result.outer[2] = res[i].outer[2];
        }
    }
    return result;
}
