#include <vector>
#include <iostream>
#include <cmath>
double zeroNormalized[16000000];
double squareNormalized[16000000];
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
constexpr double4_t d4zero = {0, 0, 0, 0};
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<double> means(ny, 0.0);
    std::vector<double> squareSums(ny, 0.0);
    int newX = (nx + 3) / 4;    
    std::vector<double4_t> normalized(ny*newX);
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
        double sum = 0;
        for (int x = 0; x < nx; x++) {
            sum += data[x+y*nx];
            
        }
        means[y] = sum / nx;
    }
    
    
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            zeroNormalized[x+y*nx] = data[x+y*nx] - means[y]; 
        }
    }
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            squareSums[y] += std::pow(zeroNormalized[x+y*nx], 2);           
        }
    }
    for (int y = 0; y<ny; y++) {
        for (int x = 0; x<newX; x++) {
            for (int vecX = 0; vecX < 4; vecX++) {
                int i = x*4+vecX;
                normalized[x + y * newX][vecX] = i < nx ? zeroNormalized[i + y * nx] / std::sqrt(squareSums[y]) : 0;
            }
        }
    }

    std::cout << "\n right stand\n";
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < newX; x++) {
            for (int i = 0; i < 4; i++) {
                std::cout << normalized[x + y * newX][i] << " ";
            }
        }
        std::cout << "\n";
    }
    
    for (int y = 0; y < ny; y++) {
        std::cout << "y: " << y << "\n";
        double4_t sum = d4zero;
        for (int x = 0; x < newX; x++) {
            sum = sum + d[x + y * newX];
        }
        std::cout << "sum: " << sum[0] << " " << sum[1] << " " << sum[2] << "\n";
        double sumDouble = 0;
        for (int i = 0; i < 4; i++) {
            sumDouble += sum[i]; 
        }
        double4_t mean = {(double)(sumDouble / nx),(double)(sumDouble / nx),(double)(sumDouble / nx),(double)(sumDouble / nx)};
        for (int x = 0; x < newX; x++) {
            d[x + y * newX] = d[x + y * newX] - mean;
        }
        for (int x = 0; x < newX; x++) {
            d[x + y * newX] = d[x + y * newX] * d[x + y * newX];
        }
    }
    std::cout << "\n vec stand\n";
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < newX; x++) {
            for (int i = 0; i < 4; i++) {
                std::cout << d[x + y * newX][i] << " ";
            }
        }
        std::cout << "\n";
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
