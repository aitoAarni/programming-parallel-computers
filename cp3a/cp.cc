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
    
    
    double4_t sum0;
    double4_t sum1;
    double total_sum0;
    double total_sum1;
    double4_t a0;
    double4_t b0;
    double4_t c0;

    double4_t a1;
    double4_t b1;
    double4_t c1;
   // std::cout << "data: \n";
   // for (int y = 0; y < ny; y++) {
   //     for (int x = 0; x < newX; x++) {
   //         for (int k = 0; k < columnBlock; k++) {
   //             std::cout << d[x + y * newX][k] << " ";
   //         }
   //     }
   //     std::cout << "\n";
   // }
   // std::cout << "nx: " << nx << "  ny: " << ny << "  newX: " << newX << "\n\n";
    // #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        for (int x = y; x < ny; x++) {
            sum0 = d4zero;
            sum1 = d4zero;
            total_sum0 = 0;
            total_sum1 = 0;
            int i = 0;
            for (; i + 1 < newX; i += 2) {
                a0 = d[i + y * newX];
                b0 = d[i + x * newX];
                c0 = a0 * b0;
                sum0 += c0;
                a1 = d[i + 1 + y * newX];
                b1 = d[i + 1 + x * newX];
                c1 = a1 * b1;
                sum1 += c1;
                
            }
            if (i < newX) {
                a0 = d[i + y * newX];
                b0 = d[i + x * newX];
                sum0 += a0 * b0;
            }
            for (int i = 0; i < columnBlock; i++) {
                total_sum0 += sum0[i];
            }
            for (int i = 0; i < columnBlock; i++) {
                total_sum1 += sum1[i];
            }
            result[x + y * ny] = total_sum0 + total_sum1;
        }
    }
}
