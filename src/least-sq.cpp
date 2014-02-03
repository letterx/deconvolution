#include "least-sq.hpp"
#include <lbfgs.h>
#include <iostream>

namespace deconvolution {

static double leastSqEvaluate(
        void *instance,
        const double *x,
        double *g,
        const int n,
        const double step) {
    return 0;
    
}

static int leastSqProgress(
        void *instance,
        const double *x,
        const double *g,
        const double fx,
        const double xnorm,
        const double gnorm,
        const double step,
        int n,
        int k,
        int ls) {
    printf("Iteration %d:\n", k);
    printf("  fx = %f", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

template <int D>
void leastSquares(const Array<D>& y, const LinearSystem<D>& H, Array<D>& x) {
    auto n = x.num_elements();
    lbfgs_parameter_t params;
    double fVal = 0;

    lbfgs_parameter_init(&params);

    auto retCode = lbfgs(n, x.origin(), &fVal, leastSqEvaluate, leastSqProgress, NULL, &params);
    std::cout << "Finished: " << retCode << "\n";
}

#define INSTANTIATE_DECONVOLVE(d) \
    template void leastSquares<d>(const Array<d>& y, const LinearSystem<d>& H, Array<d>& x);
INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
