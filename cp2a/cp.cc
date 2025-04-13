#include <vector>
#include <iostream>
#include <cmath>
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
    std:: vector<double> zeroNormalized(16000000, 0.0);   
    std::vector<double> squareNormalized(16000000, 0.0);
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
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            squareNormalized[x + y * nx] = zeroNormalized[x+y*nx] / std::sqrt(squareSums[y]);
        }
    }


    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double sum = 0;
            int x;
            for (x = 0; x + 3 < nx; x += 4) {
                sum += squareNormalized[x + nx * j] * squareNormalized[x + nx * i];
                sum += squareNormalized[x + 1 + nx * j] * squareNormalized[x + 1 + nx * i];
                sum += squareNormalized[x + 2 + nx * j] * squareNormalized[x + 2 + nx * i];
                sum += squareNormalized[x + 3 + nx * j] * squareNormalized[x + 3 + nx * i];
            }
            for (; x < nx; x++) {
                sum += squareNormalized[x + nx * j] * squareNormalized[x + nx * i];
}
            result[i + j * ny] = (float)(sum);
        }
    }
}
