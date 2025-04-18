#include <vector>
#include <iostream>
#include <cmath>
double zeroNormalized[16000000];
double squareNormalized[16000000];
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
constexpr double4_t d4zero = {0, 0, 0, 0};


static inline double4_t sqrt_vector(double4_t v) {
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
    constexpr int columnBlock = 4;
    constexpr int rowBlock = 3;
    const int newX = (nx + columnBlock - 1) / columnBlock;    
    const int newY = (ny + rowBlock - 1) / rowBlock;
    const int dataHeight = newY * rowBlock;
    std::vector<double4_t> d(newY * rowBlock * newX);
    for (int y = 0; y<dataHeight; y++) {
        for (int x = 0; x<newX; x++) {
            for (int vecX = 0; vecX < columnBlock; vecX++) {
                int i = x*columnBlock+vecX;
                d[x + y * newX][vecX] = i < nx && y < ny ? data[i + y * nx] : 0;
            }
        }
    }
    
    
    #pragma omp parallel for
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
        double4_t den = {squareSum, squareSum, squareSum, squareSum};
        double4_t denominator = sqrt_vector(den);
        for (int x = 0; x < newX; x++) {
            d[x + y * newX] = d[x + y * newX] / denominator;
        }
        for (int x = 1; x < 4; x++) {
            if (4 * (newX - 1) + x >= nx) d[( newX - 1) + y * newX][x] = 0;
        }
    }
//    std::cout << "ny: " << ny << "  nx: "<< nx << "  rowBlock: " << rowBlock << "  dataHeight: " << dataHeight << "  newY: " << newY << "\n\n";
//    for (int y = 0; y < dataHeight; y++) {
//        for (int x = 0; x < newX; x++) {
//            for (int k = 0; k < columnBlock; k++) {
//                std::cout << d[x + y * newX][k] << " ";
//            }
//        }
//        std::cout << "\n";
//    }
    
    #pragma omp parallel for
    for (int y = 0; y < newY; y++) {
        for (int x = y; x < newY; x++) {
            double total_sum;
            double4_t vv[rowBlock][2];
            double4_t sums[rowBlock][rowBlock];
            double4_t c[rowBlock][rowBlock];
            
            for (int i = 0; i < rowBlock; i++) {
                for (int j = 0; j < rowBlock; j++) {
                    sums[i][j] = d4zero;
                }
            }
            for (int i = 0; i < newX; i++) {
                int aBase = i + y * newX * rowBlock;
                int bBase = i + x * newX * rowBlock;
                // y * rowBlock * newX + i;
                vv[0][0] = d[aBase];
                vv[0][1] = d[bBase];
                vv[1][0] = d[aBase + newX];
                vv[1][1] = d[bBase + newX];
                vv[2][0] = d[aBase + newX * 2];
                vv[2][1] = d[bBase + newX * 2];
                for (int yy = 0; yy < rowBlock; yy++) {
                    for (int xx = 0; xx < rowBlock; xx++) {
                        c[yy][xx] = vv[yy][0] * vv[xx][1];
                        sums[yy][xx] += c[yy][xx];
                    }
                }
            }
            for (int yy = 0; yy < rowBlock; yy++) {
                for (int xx = 0; xx < rowBlock; xx++) {
                    total_sum = 0;
                    for (int i = 0; i < 4; i++) {
                        total_sum += sums[yy][xx][i];
                        }
                        int xCoord = x * rowBlock + xx;
                        int yCoord = y * rowBlock + yy;
                        if (xCoord < ny && yCoord < ny) {
                            result[xCoord + yCoord * ny] = total_sum;
                        }

                    }
                }
            }
    }
    
}