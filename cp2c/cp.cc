#include <vector>
#include <iostream>
#include <cmath>
double zeroNormalized[16000000];
double squareNormalized[16000000];
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
constexpr double4_t d4zero = {0, 0, 0, 0};


double4_t sqrt_vector(double4_t v) {
    double4_t result;
    for (int i = 0; i < 4; ++i) {
        result[i] = std::sqrt(v[i]);
    }
    return result;
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
    
    
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < ny; x++) {
            double4_t sum = d4zero;
            double total_sum = 0;
            for (int i = 0; i < newX; i++) {
                double4_t a = d[i + y * newX];
                double4_t b = d[i + x * newX];
                a = a * b;
                sum = sum + a;
                
            }
            for (int i = 0; i < 4; i++) {
                total_sum += sum[i];
            }
            result[x + y * ny] = total_sum;
        }
    }
}
