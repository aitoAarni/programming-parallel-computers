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

typedef double double4_t __attribute__ ((vector_size (8 * sizeof(double))));

struct ResultD {
    int y0;
    int x0;
    int y1;
    int x1;
    double4_t outer;
    double4_t inner;
};


double4_t rec_sum[400][400];
double4_t sum_square[400][400];
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    constexpr int columnBlock = 2;
    int newX = (nx + columnBlock - 1) / columnBlock;
    printf("columnBlock: %d,  newX: %d\n\n", columnBlock, newX);
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

    ResultD res[22];
    double min_thread[22];
    for (int i = 0; i < 22; i ++) {
        min_thread[i] = 10e+50;
    }
    double lowest_score = 10e+50;
    #pragma omp parallel for schedule(static, 1)
    for (int y0 = 0; y0 < ny; y0++) {
        for (int x0 = 0; x0 < nx; x0++) {
            for (int y1 = y0; y1 < ny; y1++){
                std::cout << "\n\n new \n\n";
                for (int x1 = x0 ; x1 < nx; x1 += columnBlock) {
                    std::cout << "x1: " << x1 << "\n";
                    double4_t sum[columnBlock];
                    double4_t square_sum[columnBlock];
                    double total_sse[columnBlock];
                    for (int i = 0; i < columnBlock; i++) {
                        total_sse[i] = 0;
                        sum[i] = rec_sum[y1][x1 + i];
                        square_sum[i] = sum_square[y1][x1 + i];
                    }

                    for (int i = 0; i < columnBlock; i++) {

                        if (y0 > 0) {
                            sum[i] -= rec_sum[y0-1][x1 + i];
                            square_sum[i] -= sum_square[y0-1][x1 + i];   
                        }
                        if (x0 > 0) {
                            sum[i] -= rec_sum[y1][x0-1];
                            square_sum[i] -= sum_square[y1][x0-1];   
                        }
                        if (y0>0 && x0>0) {
                            sum[i] += rec_sum[y0-1][x0-1];
                            square_sum[i] += sum_square[y0-1][x0-1];   
                        }
                    }
                    double4_t rec_size;
                    double4_t background_size;
                    for (int i = 0; i < columnBlock; i++) {
                        rec_size[i] = (x1-x0+1) * (y1-y0 + 1);
                        background_size[i] = ny * nx - rec_size[i];
                    }
                    double4_t background_sum[columnBlock];
                    double4_t background_square_sum[columnBlock];
                    double4_t rec_sse[columnBlock];
                    double4_t background_sse[columnBlock];
                    for (int i = 0; i < columnBlock; i++) {

                        
                        background_sum[i] = rec_sum[ny-1][nx-1] - sum[i];
                        background_square_sum[i] = sum_square[ny-1][nx-1] - square_sum[i];
                        rec_sse[i] = square_sum[i] - ((sum[i] * sum[i]) / (rec_size));
                        background_sse[i] = background_square_sum[i] - ((background_sum[i] * background_sum[i]) / background_size);
                        for (int j = 0; j < 3; j++) {
                            total_sse[i] += rec_sse[i][j];
                            total_sse[i] += background_sse[i][j];
                        }
                    }
                    for (int i = 0; i < columnBlock; i++) {

                        if (total_sse[i] < lowest_score) {
                            int thread = omp_get_thread_num();
                            min_thread[thread] = total_sse[i];                                                        
                            lowest_score = total_sse[i];
                            res[thread].y0 = y0;
                            res[thread].x0 = x0;
                            res[thread].y1 = y1 + 1;
                            res[thread].x1 = x1 + 1 + i;
                            res[thread].inner = sum[i] / rec_size;
                            res[thread].outer = background_sum[i] / background_size;
                        }
                    }

                }
            }

        }
    }
    double minimum = 10e+50;
    int minIndex = 0;
    for (int i = 0; i < 22; i++) {
        if (min_thread[i] < minimum) {
            minimum = min_thread[i];
            minIndex = i;
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
