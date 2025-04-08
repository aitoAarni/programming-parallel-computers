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
    for (int y = 0; y < ny; y++) {
        double sum = 0;
        for (int x = 0; x < nx; x++) {
            sum += data[x+y*nx];
        }
        means[y] = sum / nx;
    }
    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {

            double numerator = 0;
            double j_sd = 0;
            double i_sd = 0;
            for (int x = 0; x < nx; x++) {
                numerator += (data[x+j*nx] - means[j]) * (data[x+i*nx] - means[i]);
                j_sd += std::pow(data[x+j*nx] - means[j], 2);
                i_sd += std::pow(data[x+i*nx] - means[i], 2);
            }
            double denominator = std::sqrt(j_sd) * std::sqrt(i_sd);
            std::cout << "j: " << j << "  i: " << i  << "  ";
            std::cout << "corr: " << numerator / denominator << "  numerator: " << numerator << "denominator: " << denominator << "\n";
            result[i+j*ny] = (float)(numerator / denominator);
        }
    }
}
