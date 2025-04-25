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

double r_rec_sum[400][400];
double g_rec_sum[400][400];
double b_rec_sum[400][400];
double r_sum_square[400][400];
double g_sum_square[400][400];
double b_sum_square[400][400];
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    for (int y = 0; y<ny; y++) {
        for (int x = 0; x < nx; x++) {
            int baseIndex = x*3 + y*nx*3;
            double sum_r = 0;
            double sum_g = 0;
            double sum_b = 0;
            double square_sum_r = 0;
            double square_sum_g = 0;
            double square_sum_b = 0;
            if (x > 0) {
                sum_r += r_rec_sum[y][x-1];
                sum_g += g_rec_sum[y][x-1];
                sum_b += b_rec_sum[y][x-1];
                square_sum_r += r_sum_square[y][x-1];
                square_sum_g += g_sum_square[y][x-1];
                square_sum_b += b_sum_square[y][x-1];
            }
            if (y > 0) {
                sum_r += r_rec_sum[y-1][x];
                sum_g += g_rec_sum[y-1][x];
                sum_b += b_rec_sum[y-1][x];
                square_sum_r += r_sum_square[y-1][x];
                square_sum_g += g_sum_square[y-1][x];
                square_sum_b += b_sum_square[y-1][x];
                if (x > 0) {
                    sum_r -= r_rec_sum[y-1][x-1];
                    sum_g -= g_rec_sum[y-1][x-1];
                    sum_b -= b_rec_sum[y-1][x-1];
                    square_sum_r -= r_sum_square[y-1][x-1];
                    square_sum_g -= g_sum_square[y-1][x-1];
                    square_sum_b -= b_sum_square[y-1][x-1];
                }
            }
            sum_r += data[baseIndex];
            sum_g += data[baseIndex + 1];
            sum_b += data[baseIndex + 2];
            square_sum_r +=  data[baseIndex] * data[baseIndex];
            square_sum_g += data[baseIndex + 1] * data[baseIndex + 1];
            square_sum_b += data[baseIndex + 2] * data[baseIndex + 2];
            r_rec_sum[y][x] = sum_r;
            g_rec_sum[y][x] = sum_g;
            b_rec_sum[y][x] = sum_b;
            r_sum_square[y][x] = square_sum_r;
            g_sum_square[y][x] = square_sum_g;
            b_sum_square[y][x] = square_sum_b;
        }
    }
    Result res[22];
    double min_thread[22];
    for (int i = 0; i < 22; i ++) {
        min_thread[i] = 10e+50;
    }
    double lowest_score = 10e+50;
    #pragma omp parallel for
    for (int y0 = 0; y0 < ny; y0++) {
        for (int x0 = 0; x0 < nx; x0++) {
            for (int y1 = y0; y1 < ny; y1++){
                for (int x1 = x0 ; x1 < nx; x1++) {
                    double total_sse = 0;

                    double r_sum = r_rec_sum[y1][x1];
                    double r_square_sum = r_sum_square[y1][x1];
                    double g_sum = g_rec_sum[y1][x1];
                    double b_sum = b_rec_sum[y1][x1];
                    double g_square_sum = g_sum_square[y1][x1];
                    double b_square_sum = b_sum_square[y1][x1];
                    if (y0 > 0) {
                        r_sum -= r_rec_sum[y0-1][x1];
                        r_square_sum -= r_sum_square[y0-1][x1];   
                        g_sum -= g_rec_sum[y0-1][x1];
                        g_square_sum -= g_sum_square[y0-1][x1];   
                        b_sum -= b_rec_sum[y0-1][x1];
                        b_square_sum -= b_sum_square[y0-1][x1];   
                    }
                    if (x0 > 0) {
                        r_sum -= r_rec_sum[y1][x0-1];
                        r_square_sum -= r_sum_square[y1][x0-1];   
                        g_sum -= g_rec_sum[y1][x0-1];
                        g_square_sum -= g_sum_square[y1][x0-1];   
                        b_sum -= b_rec_sum[y1][x0-1];
                        b_square_sum -= b_sum_square[y1][x0-1];   
                    }
                    if (y0>0 && x0>0) {
                        r_sum += r_rec_sum[y0-1][x0-1];
                        r_square_sum += r_sum_square[y0-1][x0-1];   
                        g_sum += g_rec_sum[y0-1][x0-1];
                        g_square_sum += g_sum_square[y0-1][x0-1];   
                        b_sum += b_rec_sum[y0-1][x0-1];
                        b_square_sum += b_sum_square[y0-1][x0-1];   
                    }
                    int rec_size = (x1-x0+1) * (y1-y0 + 1);
                    int background_size = ny * nx -rec_size;
                
                    double r_background_sum = r_rec_sum[ny-1][nx-1] - r_sum;
                    double r_background_square_sum = r_sum_square[ny-1][nx-1] - r_square_sum;
                    double r_rec_sse = r_square_sum - ((r_sum * r_sum) / (rec_size));
                    double r_background_sse = r_background_square_sum - ((r_background_sum * r_background_sum) / background_size);
                    
                    
                    double g_background_sum = g_rec_sum[ny-1][nx-1] - g_sum;
                    double g_background_square_sum = g_sum_square[ny-1][nx-1] - g_square_sum;
                    double g_rec_sse = g_square_sum - ((g_sum * g_sum) / (rec_size));
                    double g_background_sse = g_background_square_sum - ((g_background_sum * g_background_sum) / background_size);

                    double b_background_sum = b_rec_sum[ny-1][nx-1] - b_sum;
                    double b_background_square_sum = b_sum_square[ny-1][nx-1] - b_square_sum;
                    double b_rec_sse = b_square_sum - ((b_sum * b_sum) / (rec_size));
                    double b_background_sse = b_background_square_sum - ((b_background_sum * b_background_sum) / background_size);


                    total_sse += r_rec_sse + r_background_sse + g_rec_sse + g_background_sse + b_rec_sse + b_background_sse;
                    if (total_sse < lowest_score) {
                            int thread = omp_get_thread_num();
                            min_thread[thread] = total_sse;                                                        
                            lowest_score = total_sse;
                            res[thread].y0 = y0;
                            res[thread].x0 = x0;
                            res[thread].y1 = y1 + 1;
                            res[thread].x1 = x1 + 1;
                            res[thread].inner[0] = r_sum / rec_size;
                            res[thread].outer[0] = r_background_sum / background_size;
                            res[thread].inner[1] = g_sum / rec_size;
                            res[thread].outer[1] = g_background_sum / background_size;
                            res[thread].inner[2] = b_sum / rec_size;
                            res[thread].outer[2] = b_background_sum / background_size;
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
            result = res[i];
        }
    }
    return result;
}
