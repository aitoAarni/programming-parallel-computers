#include "pngio.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cp.h"
#include "tests.h"

ppc::Image8 visualize(input &instance, const std::vector<float> &output, bool scalar) {
    int ny = instance.ny;
    int nx = instance.nx;
    if (!scalar) {
        nx /= 3;
    }

    ppc::Image8 result;
    result.resize(ny, nx + 2 + ny, 3);

    // draw separating lines
    for (int y = 0; y < ny; ++y) {
        if (scalar) {
            result.setlin3(y, nx, 1.0, 0.5, 0.0);
            result.setlin3(y, nx + 1, 1.0, 0.5, 0.0);
        } else {
            result.setlin3(y, nx, 1.0, 1.0, 1.0);
            result.setlin3(y, nx + 1, 0.0, 0.0, 0.0);
        }
    }

    // detect scale of inputs
    float lo = *std::min_element(instance.input.begin(), instance.input.end());
    float hi = *std::max_element(instance.input.begin(), instance.input.end());

    // draw the original image
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (scalar) {
                float c = instance.input[x + nx * y];
                if (c > 0) {
                    c = c / (hi + 1e-5f);
                    result.setlin3(y, x, 1.0, 1 - c, 1 - c);
                } else {
                    c = c / (lo - 1e-5f);
                    result.setlin3(y, x, 1 - c, 1 - c, 1.0);
                }
            } else {
                for (int c = 0; c < 3; ++c) {
                    result.setlin(y, x, c, instance.input[c + 3 * (x + nx * y)]);
                }
            }
        }
    }

    // draw the correlation
    for (int i = 0; i < ny; ++i) {
        for (int j = i; j < ny; ++j) {
            float c = output[i * ny + j];
            if (c > 0) {
                result.setlin3(i, nx + 2 + j, 1.0, 1.0 - c, 1.0 - c);
                result.setlin3(j, nx + 2 + i, 1.0, 1.0 - c, 1.0 - c);
            } else {
                result.setlin3(i, nx + 2 + j, 1.0 + c, 1.0 + c, 1.0);
                result.setlin3(j, nx + 2 + i, 1.0 + c, 1.0 + c, 1.0);
            }
        }
    }

    return result;
}

ppc::Image8 handle_test_case(const char *testfile) {
    std::ifstream input_file(testfile);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        std::exit(1);
    }

    std::string input_type;
    CHECK_READ(input_file >> input_type);
    if (input_type == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        CHECK_READ(input_file >> input_type);
    }

    input input;
    if (input_type == "raw") {
        input = generate_raw_input(input_file);
    } else if (input_type == "random") {
        input = generate_random_input(input_file);
    } else {
        std::cerr << "Invalid input type" << std::endl;
        std::exit(3);
    }

    std::vector<float> output(input.ny * input.ny);
    correlate(input.ny, input.nx, input.input.data(), output.data());

    return visualize(input, output, true);
}

ppc::Image8 handle_image(const char *testfile) {
    ppc::Image8 im_input;
    ppc::read_image(im_input, testfile);

    if (im_input.nc != 3) {
        std::cerr << "Image " << testfile << " should have 3 color components." << std::endl;
        std::exit(1);
    }

    input instance;
    instance.nx = 3 * im_input.nx;
    instance.ny = im_input.ny;
    instance.input.resize(3 * instance.nx * instance.ny);
    std::transform(im_input.data.begin(), im_input.data.end(), instance.input.begin(),
                   [](unsigned char cc) {
                       return static_cast<float>(cc) / 255.f;
                   });

    std::vector<float> output(instance.ny * instance.ny);
    correlate(instance.ny, instance.nx, instance.input.data(), output.data());
    return visualize(instance, output, false);
}

void help(const char *prog) {
    std::cout << "usage: " << prog << " test.txt [output]" << std::endl;
    std::cout << "       " << prog << " test.png [output]" << std::endl;
    std::cout << "  If test is a png image, run the correlation on that image." << std::endl;
    std::cout << "  If test is a test case, run the correlation on that test case." << std::endl;
    std::exit(1);
}

int main(int argc, const char **argv) {
    if (argc != 2 && argc != 3) {
        help(argv[0]);
    }
    const char *testfile = argv[1];
    std::size_t len = strlen(testfile);
    if (len < 4) {
        std::cerr << "Input file name too short. Please use either a '*.txt' or a '*.png' file." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const char *imfile_output = "out.png";
    if (argc == 3) {
        imfile_output = argv[2];
    }

    auto result = [&]() {
        if (strcmp(testfile + len - 4, ".png") == 0) {
            return handle_image(testfile);
        } else if (strcmp(testfile + len - 4, ".txt") == 0) {
            return handle_test_case(testfile);
        } else {
            help(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }();

    write_image(result, imfile_output);

    std::cout << "Generated result '" << imfile_output << "'\n";

    return EXIT_SUCCESS;
}
