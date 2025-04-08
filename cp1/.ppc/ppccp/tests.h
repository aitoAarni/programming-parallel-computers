#pragma once

#include "ppc.h"

struct input {
    int ny;
    int nx;
    std::vector<float> input;
};

static void generate(int ny, int nx, float *data) {
    ppc::random rng;
    for (int y = 0; y < ny; ++y) {
        if (y > 0 && rng.get_uint64(0, 1)) {
            // Introduce some correlations
            int row = std::min(static_cast<int>(y * rng.get_double()), y - 1);
            double offset = 2.0 * (rng.get_double() - 0.5f);
            double mult = 2.0 * rng.get_double();
            for (int x = 0; x < nx; ++x) {
                double error = 0.1 * rng.get_double();
                data[x + nx * y] = mult * data[x + nx * row] + offset + error;
            }
        } else {
            // Generate random data
            for (int x = 0; x < nx; ++x) {
                double v = rng.get_double();
                data[x + nx * y] = v;
            }
        }
    }
}

// Generates random rows in a random 3D subspace
static void generate_subspace(int ny, int nx, float *data) {
    ppc::random rng;

    std::vector<float> a(nx);
    std::vector<float> b(nx);
    std::vector<float> c(nx);
    for (int x = 0; x < nx; x++) {
        a[x] = rng.get_float_normal();
        b[x] = rng.get_float_normal();
        c[x] = rng.get_float_normal();
    }

    for (int y = 0; y < ny; y++) {
        float am = rng.get_float_normal();
        float bm = rng.get_float_normal();
        float cm = rng.get_float_normal();
        float len = std::sqrt(am * am + bm * bm + cm * cm);
        am /= len;
        bm /= len;
        cm /= len;
        for (int x = 0; x < nx; x++)
            data[x + nx * y] = am * a[x] + bm * b[x] + cm * c[x];
    }
}

static void generate_normal(int ny, int nx, float *data) {
    ppc::random rng;
    std::generate(data, data + nx * ny, [&] { return rng.get_float_normal(); });
}

static void generate_special(int ny, int nx, float *data) {
    ppc::random rng;
    const float a = std::numeric_limits<float>::max();
    std::generate(data, data + nx * ny, [&] { return rng.get_double(-a, a); });
}

static void generate_measurement(int ny, int nx, float *data) {
    ppc::random rng;
    std::vector<double> target(nx);
    std::generate(target.begin(), target.end(), [&] { return rng.get_double(); });
    for (int j = 0; j < ny; j++) {
        // parameters for y = ax + b
        double a = 4.0 * rng.get_double() + 0.25; // [0.25, 4.25]
        if (rng.get_uint64(0, 1))
            a = -a;
        double b = 4.0 * rng.get_double() - 2.0; // [-2.0, 2.0]
        double error = rng.get_double();
        for (int i = 0; i < nx; i++) // generate related data with error
            data[j * nx + i] = a * target[i] + b + (rng.get_double() - 0.5) * 2.0 * error;
    }
}

static void generate_benchmark(int ny, int nx, float *data) {
    ppc::random rng;
    for (int y = 0; y < ny; ++y) {
        int row_type = rng.get_int32(0, 2);
        if (row_type == 0) {
            // uniform row
            for (int x = 0; x < nx; ++x) {
                data[x + nx * y] = rng.get_float();
            }
        } else if (row_type == 1) {
            // normal row
            for (int x = 0; x < nx; ++x) {
                data[x + nx * y] = rng.get_float_normal();
            }
        } else if (row_type == 2) {
            // outlier row
            for (int x = 0; x < nx; ++x) {
                data[x + nx * y] = rng.get_float();
            }
            int o = rng.get_int32(0, nx - 1);
            data[o + nx * y] = rng.get_float(-100, 100);
        }
    }
}

static input generate_random_input(std::ifstream &input_file) {
    ppc::random rng;
    int ny, nx;
    int mode;
    CHECK_READ(input_file >> ny >> nx >> mode);
    CHECK_END(input_file);

    std::vector<float> data(ny * nx);

    switch (mode) {
    case 0:
        generate(ny, nx, data.data());
        break;
    case 1:
        generate_normal(ny, nx, data.data());
        break;
    case 2:
        generate_subspace(ny, nx, data.data());
        break;
    case 3:
        generate_measurement(ny, nx, data.data());
        break;
    case 4:
        generate_special(ny, nx, data.data());
        break;
    case 5:
        generate_benchmark(ny, nx, data.data());
        break;
    default:
        std::cerr << "unknown MODE\n";
        std::exit(3);
    }

    return {ny, nx, data};
}

static input generate_raw_input(std::ifstream &input_file) {
    int ny, nx;
    std::string header;
    CHECK_READ(getline(input_file, header));
    std::stringstream header_reader(header);
    CHECK_READ(header_reader >> ny >> nx);
    CHECK_END(header_reader);

    std::vector<float> data(ny * nx);
    for (int y = 0; y < ny; y++) {
        std::string line;
        CHECK_READ(getline(input_file, line));
        std::stringstream line_reader(line);
        for (int x = 0; x < nx; x++) {
            float value;
            CHECK_READ(line_reader >> value);
            data[y * nx + x] = value;
        }
        CHECK_END(line_reader);
    }
    CHECK_END(input_file);

    return {ny, nx, data};
}
