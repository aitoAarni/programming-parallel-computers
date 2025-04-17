#include <vector>
#include <iostream>
#include <cmath>
double zeroNormalized[16000000];
double squareNormalized[16000000];
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
constexpr double4_t d4zero = {0, 0, 0, 0};


double4_t sqrt_vector(double4_t v) {
    double4_t r = d4zero;
    for (int i = 0; i < 4; ++i) {
        r[i] = std::sqrt(v[i]);
    }
    return r;
}
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < x
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    int newX = (nx + 3) / 4;    
    constexpr int columnBlock = 4;
    std::vector<double4_t> d(ny * newX);
    for (int y = 0; y<ny; y++) {
        for (int x = 0; x<newX; x++) {
            for (int vecX = 0; vecX < 4; vecX++) {
                int i = x*4+vecX;
                d[x + y * newX][vecX] = i < nx ? data[i + y * nx] : 0;
            }
        }
    }
    
    
    
    for (int y = 0; y < ny; y++) {
        double4_t sum = d4zero;
        for (int x = 0; x < newX; x++) {
            sum = sum + d[x + y * newX];
        }
        double sumDouble = 0;
        for (int i = 0; i < 4; i++) {
            sumDouble += sum[i]; 
        }
        double4_t mean = {(double)(sumDouble / nx),(double)(sumDouble / nx),(double)(sumDouble / nx),(double)(sumDouble / nx)};
        for (int x = 0; x < newX; x++) {
            d[x + y * newX] = d[x + y * newX] - mean;
        }
        for (int x = 1; x < 4; x++) {
            if (4 * (newX - 1) + x >= nx) {
                d[(newX - 1) + y * newX][x] = 0;
            }
        }
        double4_t squareSums = d4zero;
        for (int x = 0; x < newX; x++) {
            double4_t a = d[x + y * newX] * d[x + y * newX];
            squareSums = squareSums + a;
        }
        double squareSum = 0;
        for (int i = 0; i < 4; i++) {
            squareSum += squareSums[i];
        }
        double4_t denominator = {squareSum, squareSum, squareSum, squareSum};
        for (int x = 0; x < newX; x++) {
            d[x + y * newX] = d[x + y * newX] / sqrt_vector(denominator);
        }
        for (int x = 1; x < 4; x++) {
            if (4 * (newX - 1) + x >= nx) d[( newX - 1) + y * newX][x] = 0;
        }
    }
    
    constexpr int p = 4;    
    double total_sum;
    double4_t sum0;
    double4_t sum1;
    double4_t sum2;
    double4_t sum3;
    double4_t a0;
    double4_t b0;
    double4_t c0;

    double4_t a1;
    double4_t b1;
    double4_t c1;
    double4_t a2;
    double4_t b2;
    double4_t c2;
    double4_t a3;
    double4_t b3;
    double4_t c3;
    double4_t vv[4][4];
    for (int y = 0; y < ny; y++) {
        for (int x = y; x < ny; x++) {
            total_sum = 0;
            for (int k = 0; k < p; k++) {
                vv[k][3] = d4zero;
            }
            int i = 0;
            for (; i + p - 1 < newX; i += p) {
                for (int k = 0; k < p; k++){
                    vv[k][0] = d[i + k + y * newX];
                    vv[k][1] = d[i + k + x * newX];
                    vv[k][2] = vv[k][0] * vv[k][1];
                    vv[k][3] += vv[k][2];
                }
                
            }
            for (int k = 0; k < p - 1; k++) {
                if (i + k < newX) {
                    vv[0][0] = d[i + k + y * newX];
                    vv[0][1] = d[i + k + x * newX];
                    vv[0][3] += vv[0][0] * vv[0][1];
                }
            }
            for (int k = 1; k < p; k++) {
                vv[0][3] += vv[k][3];
            }
            for (int i = 0; i < columnBlock; i++) {
                total_sum += vv[0][3][i];
            }
            result[x + y * ny] = total_sum;
        }
    }
}
