#include "least-sq.hpp"
#include "common.hpp"
#include "util.hpp"
#include <lbfgs.h>
#include <iostream>

namespace deconvolution {

template <int D>
struct LeastSquareData {
    const LinearSystem<D>& Q;
    const Array<D>& Ht_y;
    const Array<D>& x;
};

template <int D>
static double leastSqEvaluate(
        void *instance,
        const double *xData,
        double *g,
        const int n,
        const double step) {
    const auto* data = static_cast<LeastSquareData<D>*>(instance);
    const auto& Q = data->Q;
    const auto& Ht_y = data->Ht_y;
    const auto& x = data->x;
    ASSERT(x.data() == xData);
    ASSERT(int(x.num_elements()) == n);

    auto Qx = Q(x);
    auto xQx = dot(x, Qx);
    auto xHt_y = dot(x, Ht_y);
    for (int i = 0; i < n; ++i)
        g[i] = 2*(Qx.data()[i] - Ht_y.data()[i]);

    // 1/x-barrier [0, 255]
    double barrier = 0.0;
    /*
    const double scale = 1.0;
    const double threshold = 0.999999999;
    const double eps = 10.0;
    for (int i = 0; i < n; ++i) {
        auto val = xData[i];
        if (val <= -eps*threshold) val = -eps*threshold;
        if (val >= 255.0+eps*threshold) val = 255.0+eps*threshold;
        barrier += scale*(1.0/(val+eps) + 1.0/(255.0-val+eps));
        g[i] += scale*(-1.0/((val+eps)*(val+eps)) + 1.0/((255.0-val+eps)*(255.0-val+eps)));
    }
    */
    std::cout << "Evaluate: " << xQx - 2*xHt_y + barrier << "\n";
    return xQx - 2*xHt_y + barrier;
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

/*
 * Solve a least squares problem min |y - Hx|
 * This is a quadratic program x^TH^THx - 2x^TH^Ty + y^Ty
 * Inputs are Q = H^TH, and H^Ty
 */
template <int D>
void leastSquares(const LinearSystem<D>& Q, const Array<D>& Ht_y, Array<D>& x) {
    auto n = x.num_elements();
    lbfgs_parameter_t params;
    double fVal = 0;

    lbfgs_parameter_init(&params);
    LeastSquareData<D> algData = {Q, Ht_y, x};

    auto retCode = lbfgs(n, x.origin(), &fVal, leastSqEvaluate<D>, leastSqProgress, &algData, &params);
    std::cout << "Finished: " << retCode << "\n";
}

#define INSTANTIATE_DECONVOLVE(d) \
    template void leastSquares<d>(const LinearSystem<d>& Q, const Array<d>& Ht_y, Array<d>& x);
INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
