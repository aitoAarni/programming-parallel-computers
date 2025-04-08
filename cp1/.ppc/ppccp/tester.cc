#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <type_traits>
#include <unistd.h>

#include "cp.h"
#include "ppc.h"
#include "tests.h"

static float verify(const input &input, const float *result, float *errors, bool randomize) {
    bool nans = false;
    double worst = 0.0;

    std::vector<pfloat> normalized(input.ny * input.nx);

    for (int j = 0; j < input.ny; j++) {
        pfloat s = 0.0;
        pfloat ss = 0.0;
        for (int i = 0; i < input.nx; i++) {
            pfloat x = input.input[j * input.nx + i];
            s += x;
            ss += x * x;
        }
        pfloat mean = s / input.nx;
        pfloat mult = 1.0 / (pfloat)std::sqrt((long double)(ss - s * mean));
        for (int i = 0; i < input.nx; i++)
            normalized[j * input.nx + i] = (input.input[j * input.nx + i] - mean) * mult;
    }

    // test that the result at (j, i) is correct
    auto test_one = [&](int j, int i) {
        float q = result[i + input.ny * j];
        if (std::isnan(q)) {
            nans = true;
            errors[i + input.ny * j] = std::numeric_limits<float>::quiet_NaN();
            return;
        }
        pfloat temp = 0.0;
        for (int x = 0; x < input.nx; ++x) {
            pfloat a = normalized[x + input.nx * i];
            pfloat b = normalized[x + input.nx * j];
            temp += a * b;
        }
        pfloat err = q - temp;
        double abserr = std::abs((double)err);
        errors[i + input.ny * j] = abserr;

        worst = std::max(abserr, worst);
    };

    ppc::random rng;
    for (int j = 0; j < input.ny; ++j) {
        if (randomize) {
            int i = rng.get_int32(j, input.ny - 1);
            test_one(j, i);
        } else {
            for (int i = j; i < input.ny; ++i) {
                test_one(j, i);
            }
        }
    }
    if (nans)
        worst = std::numeric_limits<float>::quiet_NaN();

    return worst;
}

// Does 'iter' iterations of Freivald's algorithm and returns the largest
// difference over all vector elements and iterations.
static float verify_gvfa(const input &input, const float *result, int iter) {
    std::vector<double> normalized(input.ny * input.nx);

    for (int j = 0; j < input.ny; j++) {
        double s = 0.0;
        for (int i = 0; i < input.nx; i++) {
            double x = input.input[j * input.nx + i];
            s += x;
        }
        double mean = s / input.nx;
        double ss = 0.0;
        for (int i = 0; i < input.nx; i++) {
            double x = input.input[j * input.nx + i];
            x -= mean;
            normalized[j * input.nx + i] = x;
            ss += x * x;
        }
        double mult = 1.0 / std::sqrt(ss);
        for (int i = 0; i < input.nx; i++)
            normalized[j * input.nx + i] *= mult;
    }

    ppc::random rng;
    std::vector<double> x(input.ny * iter);
    std::vector<double> ATx(input.nx * iter, 0.0);
    std::vector<double> AATx(input.ny * iter, 0.0);
    std::vector<double> Bx(input.ny * iter);
    for (int j = 0; j < input.ny; j++) {
        for (int k = 0; k < iter; k++) {
            x[j * iter + k] = rng.get_double_normal();
        }
    }

    for (int j = 0; j < input.ny; j++) {
        for (int i = 0; i < input.nx; i++) {
            double left = normalized[j * input.nx + i];
            for (int k = 0; k < iter; k++)
                ATx[i * iter + k] += left * x[j * iter + k];
        }
    }

    for (int j = 0; j < input.ny; j++) {
        for (int i = 0; i < input.nx; i++) {
            for (int k = 0; k < iter; k++)
                AATx[j * iter + k] += normalized[j * input.nx + i] * ATx[i * iter + k];
        }
    }

    for (int j = 0; j < input.ny; j++) {
        for (int i = j; i < input.ny; i++) {
            double left = result[j * input.ny + i];
            for (int k = 0; k < iter; k++)
                Bx[j * iter + k] += left * x[i * iter + k];
            if (i != j) {
                for (int k = 0; k < iter; k++)
                    Bx[i * iter + k] += left * x[j * iter + k];
            }
        }
    }

    float worst = 0.0f;
    for (int j = 0; j < input.ny; j++) {
        for (int k = 0; k < iter; k++) {
            float diff = AATx[j * iter + k] - Bx[j * iter + k];
            float err = std::abs(diff);
            if (err != err)
                return err; // NaN
            worst = std::max(err, worst);
        }
    }
    return worst;
}

int main(int argc, const char **argv) {
    const char *ppc_output = std::getenv("PPC_OUTPUT");
    int ppc_output_fd = 0;
    if (ppc_output) {
        ppc_output_fd = std::stoi(ppc_output);
    }
    if (ppc_output_fd <= 0) {
        ppc_output_fd = 1;
    }
    std::unique_ptr<ppc::fdostream> stream = std::unique_ptr<ppc::fdostream>(new ppc::fdostream(ppc_output_fd));

    argc--;
    argv++;
    if (argc < 1 || argc > 2) {
        std::cerr << "Invalid usage" << std::endl;
        return 1;
    }

    bool test = false;
    if (argv[0] == std::string("--test")) {
        test = true;
        argc--;
        argv++;
    }

    const float allowed_error = PPC_ALLOW_FLOAT
                                    ? 1e-5
                                    : std::numeric_limits<float>::epsilon() * 0.6;
    constexpr float gvfa_limit = 1e-3;

    std::ifstream input_file(argv[0]);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        return 2;
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
        return 3;
    }

    std::vector<float> output(input.ny * input.ny);

    // ensure that `output` is initialized by non-zero numbers
    ppc::random rng;
    std::generate(begin(output), end(output), [&]() { return rng.get_double(); });
    // make a copy of our inputs, so our tests are still reliable if they get overwritten "accidentally"
    std::vector<float> data_copy = input.input;

    ppc::setup_cuda_device();
    ppc::perf timer;
    timer.start();
    correlate(input.ny, input.nx, data_copy.data(), output.data());
    timer.stop();
    timer.print_to(*stream);
    ppc::reset_cuda_device();

    *stream << "ny\t" << input.ny << '\n'
            << "nx\t" << input.nx << '\n';

    if (test) {
        std::vector<float> errors(input.ny * input.ny);
        float gvfa_error = verify_gvfa(input, output.data(), 20);
        bool pass = gvfa_error < gvfa_limit;
        bool spot_check = (double)input.nx * input.ny * input.ny > 1e8;
        float max_error = verify(input, output.data(), errors.data(), spot_check);
        if (spot_check) {
            // if we only do a spot check, it is possible we missed some error that
            // gets caught by gvfa
            pass = pass && max_error < allowed_error;
        } else {
            bool full_pass = max_error < allowed_error;
            // gvfa should never claim error if the result is actually ok
            assert(pass || !full_pass);
            pass = full_pass;
        }

        if (!pass) {
            bool small = input.ny * input.nx <= 200 && input.ny * input.ny <= 200;
            stream->precision(std::numeric_limits<float>::max_digits10 - 1);
            *stream
                << "result\tfail\n"
                << "gvfa_error\t" << std::scientific << gvfa_error << '\n'
                << "gvfa_error_limit\t" << std::scientific << gvfa_limit << '\n'
                << "max_error\t" << std::scientific << max_error << '\n'
                << "max_error_limit\t" << std::scientific << allowed_error << '\n';
            *stream << "size\t" << (small ? "small" : "large") << '\n';

            if (small) {
                *stream << "input\t";
                ppc::print_matrix(input.ny, input.nx, input.input.data(), stream);
                *stream << '\n';

                *stream << "output\t";
                ppc::print_matrix(input.ny, input.ny, output.data(), stream);
                *stream << '\n';

                *stream << "locations\t";
                ppc::print_matrix(input.ny, input.ny, errors.data(), stream);
                *stream << '\n';
            }
        } else {
            *stream << "result\tpass\n";
        }
    } else {
        *stream << "result\tdone\n";
    }
    *stream << std::endl;
}
