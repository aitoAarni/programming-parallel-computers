#include <vector>
#include <iostream>
#include <cmath>
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
typedef float float8_t __attribute__ (( vector_size (8 * sizeof(float))));
constexpr double4_t d4zero = {0, 0, 0, 0};
constexpr float8_t f8zero = {0, 0, 0, 0, 0, 0, 0, 0};


static inline float8_t sqrt_vector(float8_t v) {
    float8_t r = f8zero;
    for (int i = 0; i < 8; ++i) {
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
    constexpr int columnBlock = 8;
    constexpr int rowBlock = 1;
    const int newX = (nx + columnBlock - 1) / columnBlock;    
    const int newY = (ny + rowBlock - 1) / rowBlock;
    const int dataHeight = newY * rowBlock;
    std::vector<float8_t> d(newY * rowBlock * newX);
    //std::vector<double4_t> d(newY * rowBlock * newX);

    #pragma omp parallel for
    for (int y = 0; y<dataHeight; y++) {
        for (int x = 0; x<newX; x++) {
            for (int vecX = 0; vecX < columnBlock; vecX++) {
                int i = x*columnBlock+vecX;
                d[x + y * newX][vecX] = i < nx && y < ny ? data[i + y * nx] : 0;
            }
        }
    }
    
    std::cout << "table after creation: \n";
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < newX; x++) {
            for (int i = 0; i < columnBlock; i++) {
                std::cout << d[x + y * newX][i] << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n\n"; 
    // #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        std::cout << "y = " << y << "\n";
        float8_t sum = f8zero;
        for (int x = 0; x < newX; x++) {
            sum += d[x + y * newX];
        }
        float sumDouble = 0;
        for (int i = 0; i < columnBlock; i++) {
            sumDouble += sum[i]; 
        }
        std::cout << "row " << y << " sum = " << sumDouble << "\n";
        std::cout << "\n Before normalization \n";
        for (int x = 0; x < newX; x++) {
            for (int i = 0; i < columnBlock; i++) {
                std::cout << d[x + y * newX][i] << " ";
            }
        }
        float8_t mean;
        for (int i = 0; i < columnBlock; i++) {
            mean[i] = (float)(sumDouble / nx);
        }
        for (int x = 0; x < newX; x++) {
            d[x + y * newX] = d[x + y * newX] - mean;
        }
        //        std::cout << "\n normalized to zero now: \n";
        //        for (int x = 0; x < newX; x++) {
            //            for (int i = 0; i < columnBlock; i++) {
                //                std::cout << d[x + ny * newX][i] << " ";
                //            }
                //        }
                for (int x = 1; x < columnBlock; x++) {
                    if (columnBlock * (newX - 1) + x >= nx) {
                        d[(newX - 1) + y * newX][x] = 0;
                    }
                }
                std::cout << "\n after normalizeation \n";
                for (int x = 0; x < newX; x++) {
                    for (int i = 0; i < columnBlock; i++) {
                        std::cout << d[x + y * newX][i] << " ";
                    }
                }
        float8_t squareSums = f8zero;
        for (int x = 0; x < newX; x++) {
            float8_t a = d[x + y * newX] * d[x + y * newX];
            squareSums = squareSums + a;
        }
        float squareSum = 0;
        for (int i = 0; i < columnBlock; i++) {
            squareSum += squareSums[i];
        }
        float8_t den;
        for (int i = 0; i < columnBlock; i++) {
            den[i] = squareSum;
        }
        float8_t denominator = sqrt_vector(den);
        std::cout << "\n Before normalization nro 2\n";
        for (int x = 0; x < newX; x++) {
            for (int i = 0; i < columnBlock; i++) {
                std::cout << d[x + y * newX][i] << " ";
            }
        }
        std::cout << "\n\n"; 
        for (int x = 0; x < newX; x++) {
            d[x + y * newX] = d[x + y * newX] / denominator;
        }
        for (int x = 1; x < columnBlock; x++) {
            if (columnBlock * (newX - 1) + x >= nx) d[( newX - 1) + y * newX][x] = 0;
        }
        std::cout << "\n After normalization nro 2\n";
        for (int x = 0; x < newX; x++) {
            for (int i = 0; i < columnBlock; i++) {
                std::cout << d[x + y * newX][i] << " ";
            }
        }
    }
    std::cout << "\n\n"; 

    std::cout << "table after everything \n";
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < newX; x++) {
            for (int i = 0; i < columnBlock; i++) {
                std::cout << d[x + y * newX][i] << " ";
            }
        }
        std::cout << "\n";
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < newY; y++) {
        for (int x = y; x < newY; x++) {
            float total_sum[rowBlock][rowBlock];
            float8_t vv[rowBlock][2];
            float8_t sums[rowBlock][rowBlock];
            float8_t c[rowBlock][rowBlock];
            
            for (int i = 0; i < rowBlock; i++) {
                for (int j = 0; j < rowBlock; j++) {
                    sums[i][j] = f8zero;
                    total_sum[i][j] = 0;
                }
            }
            for (int i = 0; i < newX; i++) {
                int aBase = i + y * newX * rowBlock;
                int bBase = i + x * newX * rowBlock;
                for (int o = 0; o < rowBlock; o++) {
                    vv[o][0] = d[aBase + newX * o];
                    vv[o][1] = d[bBase + newX * o];
                }
                for (int yy = 0; yy < rowBlock; yy++) {
                    for (int xx = 0; xx < rowBlock; xx++) {
                        sums[yy][xx] = sums[yy][xx] + vv[yy][0] * vv[xx][1];
                    }
                }
            }
            for (int yy = 0; yy < rowBlock; yy++) {
                for (int xx = 0; xx < rowBlock; xx++) {
                     for (int i = 0; i < columnBlock; i++) {
                         total_sum[yy][xx] += sums[yy][xx][i];
                         }
                        int xCoord = x * rowBlock + xx;
                        int yCoord = y * rowBlock + yy;
                        if (xCoord < ny && yCoord < ny) {
                            result[xCoord + yCoord * ny] = total_sum[yy][xx];
                        }

                    }
                }
            }
    }
}