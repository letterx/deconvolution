#include "quadratic-min.hpp"
#include "common.hpp"
#include "util.hpp"
#include <lbfgs.h>
#include <iostream>

namespace deconvolution {

template <int D>
struct QuadraticMinData {
    const LinearSystem<D>& Q;
    const Array<D>& b;
    const Array<D>& x;
};

template <int D>
static double quadraticEvaluate(
        void *instance,
        const double *xData,
        double *g,
        const int n,
        const double step) {
    const auto* data = static_cast<QuadraticMinData<D>*>(instance);
    const auto& Q = data->Q;
    const auto& b = data->b;
    const auto& x = data->x;
    ASSERT(x.data() == xData);
    ASSERT(int(x.num_elements()) == n);

    auto Qx = Q(x);
    auto xQx = dot(x, Qx);
    auto bx = dot(b, x);
    for (int i = 0; i < n; ++i)
        g[i] = 2*Qx.data()[i] - b.data()[i];

    std::cout << "Evaluate: " << xQx - bx << "\n";
    return xQx - bx;
}

static int quadraticProgress(
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
    printf("\tIteration %d:\n", k);
    printf("\t  fx = %f", fx);
    printf("\t  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

template <int D>
double quadraticMin(const LinearSystem<D>& Q, const Array<D>& b, Array<D>& x) {
    auto n = x.num_elements();
    lbfgs_parameter_t params;
    double fVal = 0;

    lbfgs_parameter_init(&params);
    QuadraticMinData<D> algData = {Q, b, x};

    auto retCode = lbfgs(n, x.origin(), &fVal, quadraticEvaluate<D>, quadraticProgress, &algData, &params);
    std::cout << "\tFinished quadraticMin: " << retCode << ",\t" << fVal << "\n";
    return fVal;
}

#define INSTANTIATE_DECONVOLVE(d) \
    template double quadraticMin<d>(const LinearSystem<d>& Q, const Array<d>& b, Array<d>& x);
INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
