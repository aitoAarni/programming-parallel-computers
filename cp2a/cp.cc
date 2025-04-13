#include <vector>
#include <iostream>
#include <cmath>
double normalized[16000000];
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    
    for (int y = 0; y < ny; y++) {
        double sum = 0;
        for (int x = 0; x < nx; x++) {
            sum += data[x+y*nx];
        }
        double mean = sum / nx;
        for (int x = 0; x < nx; x++) {
           normalized[x+y*nx] = data[x+y*nx] - mean;  
        }
        double squareSum = 0;
        for (int x = 0; x < nx; x++) {
            squareSum += std::pow(normalized[x+y*nx], 2);           
        }
        for (int x = 0; x < nx; x++) {
            normalized[x + y * nx] = normalized[x+y*nx] / std::sqrt(squareSum);
        }
    }
    

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double sum1 = 0;
            double sum2 = 0;
            int x = 0;
            for (; x + 1 < nx; x += 2) {
                double a0 = normalized[x + nx * j];
                double b0 = normalized[x + nx * i];
                double a1 = normalized[x + 1 + nx * j];
                double b1 = normalized[x + 1 + nx * i];
                
                sum1 += a0 * b0;
                sum2 += a1 * b1;
            }
            if (x < nx) {
                sum1 += normalized[x + j * nx] * normalized[x + i * nx];
            }
            result[i + j * ny] = sum1 + sum2;
        }
    }
}
