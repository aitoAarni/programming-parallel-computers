#include <vector>
#include <iostream>
#include <cmath>
double zeroNormalized[16000000];
double squareNormalized[16000000];
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
    for (int y = 0; y < ny; y++) {
        double sum = 0;
        std::cout << "row: " << y << "\n";
        for (int x = 0; x < nx; x++) {
            std::cout << data[x+y*nx] << " ";
            sum += data[x+y*nx];
        }
        means[y] = sum / nx;
        std::cout << "mean: " << means[y] << "\n\n";
    }
    
    std::cout << "normalized \n"; 
    for (int y = 0; y < ny; y++) {
        std::cout << "row: " << y << "\n";
        for (int x = 0; x < nx; x++) {
           zeroNormalized[x+y*nx] = data[x+y*nx] - means[y]; 
           std::cout << zeroNormalized[x+y*nx] << " ";
        }
        std::cout << "\n";
    }
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            squareSums[y] += std::pow(zeroNormalized[x+y*nx], 2);           
        }
    }
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            squareNormalized[y] = zeroNormalized[x+y*nx] - (squareSums[y] - 1);
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
                result[i + nx*j] = 0;
            for (int x = 0; x < nx; x++) {
                result[i + nx*j] += squareNormalized[x + nx * j] * squareNormalized[x+ nx * i];
            }
        }
    }
}
