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
            if (x > 0) {
                sum_r += r_rec_sum[y][x-1];
            }
            if (y > 0) {
                sum_r += r_rec_sum[y-1][x];
                if (x > 0) sum_r -= r_rec_sum[y-1][x-1];
            }
            sum_r += data[baseIndex];
            r_rec_sum[y][x] = sum_r;
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