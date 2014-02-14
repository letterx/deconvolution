#include "deconvolve.hpp"
#include "least-sq.hpp"
#include "util.hpp"
#include "regularizer.hpp"
#include <lbfgs.h>

namespace deconvolution{

template <int D>
struct DeconvolveData {

};

template <int D>
static double deconvolveEvaluate(
        void* instance,
        const double* dualVars, 
        double* grad,
        const int n,
        const double step) {
    return 0;
}

static int deconvolveProgress(
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
    printf("Deconvolve Iteration %d:\n", k);
    printf("  fx = %f", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

template <int D>
Array<D> Deconvolve(const Array<D>& y, const LinearSystem<D>& H, const LinearSystem<D>& Ht, const Regularizer<D>& R) {
    Array<D> Ht_y = Ht(y);
    Array<D> x = Ht_y;
    int numPrimalVars = x.num_elements();
    int numLambda = numPrimalVars*R.numSubproblems()*R.numLabels();
    int numDualVars = numLambda + numPrimalVars; // Including all lambda vars + vector nu
    auto dualVars = std::unique_ptr<double>(new double[numDualVars]);
    double* lambda = dualVars.get();
    double* nu = dualVars.get() + numLambda;
    for (int i = 0; i < numLambda; ++i)
        lambda[i] = 0;
    for (int i = 0; i < numPrimalVars; ++i)
        nu[i] = 0;
    LinearSystem<D> Q = [&](const Array<D>& x) -> Array<D> { return Ht(H(x)) + 0.03*x; };
    for (size_t i = 0; i < x.num_elements(); ++i) {
        x.data()[i] = 0;
    }
    leastSquares<D>(Q, Ht_y, x);

    lbfgs_parameter_t params;
    double fVal = 0;
    lbfgs_parameter_init(&params);
    auto algData = DeconvolveData<D>{};
    auto retCode = lbfgs(numDualVars, dualVars.get(), &fVal, deconvolveEvaluate<D>, deconvolveProgress, &algData, &params);
    std::cout << "Deconvolve finished: " << retCode << "\n";

    return x;
}

#define INSTANTIATE_DECONVOLVE(d) \
    template Array<d> Deconvolve<d>(const Array<d>& y, const LinearSystem<d>& H, const LinearSystem<d>& Q, const Regularizer<d>& R);
INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
