#include <iostream>
#include <segment.h>

int main() {

    float data2[] = {
        0.5, 0.3, 0.7,
        0.5, 0.3, 0.7,
        0.2, 0.0, 0.0,
        0.5, 0.3, 0.7,
        0.5, 0.3, 0.7,
        0.2, 0.0, 0.0,
        0.2, 0.0, 0.0,
        0.2, 0.0, 0.0,
        0.2, 0.0, 0.0,
    };

    Result res = segment(3, 3, data2);

    std::cout << "Result rectangle: (" 
              << res.y0 << "," << res.x0 << ") -> (" 
              << res.y1 << "," << res.x1 << ")\n";

    std::cout << "Outer color: [" 
              << res.outer[0] << ", " << res.outer[1] << ", " << res.outer[2] << "]\n";

    std::cout << "Inner color: [" 
              << res.inner[0] << ", " << res.inner[1] << ", " << res.inner[2] << "]\n";

    return 0;
}