#include <iostream>
#include <vector>
#include <algorithm>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      std::vector<int> range;
      std::cout << "\n\n\n" << x << " " << y << "\n";
      for (int j = y - hy; j <= y + hy && j < ny; j++) {
        if (j < 0) continue;
        for (int i = x - hx; i <= x + hx && i < nx; i++) {
          if (i < 0) continue;
          range.push_back(in[i+nx*j]);
          std::cout << in[i+nx*j] << " ";
        }
      }
      std::cout << "\n";
      double median;
      if (range.size() % 2 == 1) {
        int mid = range.size() / 2;
        std::nth_element(range.begin(), range.begin() + mid, range.end());
        median = range[mid];
      } else {
        int mid = range.size() / 2 - 1;
        std::nth_element(range.begin(), range.begin() +  mid, range.end());
        median = (range[mid] + range[mid+1]) / (double)2;
      }
      std::cout << "median: " << median;
    }
  }
}

int main() {
  float in[9] = {3, 4, 5, 6, 7, 8, 9, 10, 11};
  float out[9];

  mf(3, 3, 1, 2, in, out);

  return 0;
}