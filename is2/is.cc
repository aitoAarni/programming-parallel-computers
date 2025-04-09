#include <iostream>
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

double r_rec_sum[50][50];
double g_rec_sum[50][50];
double b_rec_sum[50][50];
double r_sum_square[50][50];
double g_sum_square[50][50];
double b_sum_square[50][50];
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    r_rec_sum[0][0] = data[0];
    for (int y = 0; y<ny; y++) {
        std::cout << "\n";
        for (int x = 0; x < nx; x++) {
            int baseIndex = x*3 + y*nx*3;
            std::cout << data[baseIndex] << " ";
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
                    square_sum_r += r_sum_square[y-1][x-1];
                    square_sum_g += g_sum_square[y-1][x-1];
                    square_sum_b += b_sum_square[y-1][x-1];
                }
            }
            sum_r += data[baseIndex];
            sum_g += data[baseIndex + 1];
            sum_b += data[baseIndex + 2];
            square_sum_r += data[baseIndex] * data[baseIndex];
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
    std::cout << "\n\rectangles\n";
    for (int y = 0; y<ny; y++) {
        std::cout << "\n";
        for (int x = 0; x < nx; x++) {
            std::cout << r_rec_sum[y][x] << " ";
        }
    }
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    return result;
}