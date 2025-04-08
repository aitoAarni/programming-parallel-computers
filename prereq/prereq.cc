#include <iostream>


struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    double r_total = 0;
    double g_total = 0;
    double b_total = 0;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x< x1; x++) {
            int base = 3*x + 3*nx*y;
            r_total += data[base];
            g_total += data[base + 1];
            b_total += data[base + 2];
        }
    } 
    int count = (y1-y0) * (x1-x0);
    Result result{{(float)(r_total / count), (float)(g_total / count), (float)(b_total / count)}};
    return result;
}
