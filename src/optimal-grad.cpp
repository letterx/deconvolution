#include "optimal-grad.hpp"
#include <vector>
#include <iostream>
#include <cmath>

namespace deconvolution {
int optimalGradDescent(int n, double *x_initial, double *fval, lbfgs_evaluate_t eval, lbfgs_progress_t prog, void *instance) {
    double L = 1.0;
    std::vector<double> grad(n, 0);
    std::vector<double> x(x_initial, x_initial+n);
    std::vector<double> next_x(x);

    double lastFVal = eval(instance, x.data(), grad.data(), n, 1.0);

    for (int iter = 0; iter < 100; ++iter) {
        double gradNorm = 0;
        for (int i = 0; i < n; ++i) gradNorm += grad[i]*grad[i];
        gradNorm = sqrt(gradNorm);
        prog(instance, x.data(), grad.data(), lastFVal, 0.0, gradNorm, L, n, iter, 1);
        for (int i = 0; i < n; ++i) 
            next_x[i] = x[i] - grad[i]/L;
        double fVal = eval(instance, next_x.data(), grad.data(), n, 1.0);
        if (fVal < lastFVal) {
            L /= 2;
            lastFVal = fVal;
            std::swap(x, next_x);
            continue;
        } else {
            L *= 2;
            std::cout << "Increasing L: " << L << "\n";
        }


    }

    return 0; 
}
}
