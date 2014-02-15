#include "deconvolve.hpp"
#include "quadratic-min.hpp"
#include "util.hpp"
#include "regularizer.hpp"
#include <lbfgs.h>
#include <limits>
#include <iostream>

namespace deconvolution{

template <int D>
struct DeconvolveData {
    Array<D>& x;
    const Array<D>& b;
    const LinearSystem<D>& Q;
    const Regularizer<D>& R;
    int numLambda;
};

template <int D>
static double deconvolveEvaluate(
        void* instance,
        const double* dualVars, 
        double* grad,
        const int n,
        const double step) {
    auto* data = static_cast<DeconvolveData<D>*>(instance);
    auto& x = data->x;
    const auto& b = data->b;
    const auto& Q = data->Q;
    const auto& R = data->R;
    const auto numPrimalVars = int(x.num_elements());
    const auto numSubproblems = R.numSubproblems();
    const auto numLabels = R.numLabels();
    const auto numPerSubproblem = numPrimalVars*numLabels;
    const auto numLambda = data->numLambda;
    const double* nu = dualVars + numLambda;
    const double t = 1;
    auto bPlusNu = b;
    for (int i = 0; i < numPrimalVars; ++i)
        bPlusNu.data()[i] += nu[i];

    for (int i = 0; i < n; ++i)
        grad[i] = 0;

    double regularizerObjective = 0;
    for (int i = 0; i < R.numSubproblems(); ++i)
        regularizerObjective += R.evaluate(i, dualVars+i*numPerSubproblem, t, grad+i*numPerSubproblem);

    double dataObjective = quadraticMin<D>(Q, bPlusNu, x);
    for (int i = 0; i < numPrimalVars; ++i)
        grad[i+numLambda] = -x.data()[i];

    double unaryObjective = 0;
    auto minTable = std::vector<double>(numLabels, 0);
    for (int i = 0; i < numPrimalVars; ++i) {
        auto nu_i = nu[i];
        double minValue = std::numeric_limits<double>::max();
        for (int xi = 0; xi < numLabels; ++xi) {
            double lambdaSum = 0;
            for (int alpha = 0; alpha < numSubproblems; ++alpha) 
                lambdaSum += dualVars[alpha*numPerSubproblem+i*numLabels+xi];
            minTable[xi] = lambdaSum + nu_i*R.getLabel(i, xi);
            minValue = std::min(minValue, minTable[xi]);
        }
        double expSum = 0;
        double nuGrad = 0;
        for (int xi = 0; xi < numLabels; ++xi) {
            minTable[xi] = exp(-(minTable[xi]-minValue)/t);
            expSum += minTable[xi];
            nuGrad += minTable[xi]*R.getLabel(i, xi);
        }
        unaryObjective += minValue - t*log(expSum);
        grad[i+numLambda] += nuGrad/expSum;
        for (int alpha = 0; alpha < numSubproblems; ++alpha)
            for (int xi = 0; xi < numLabels; ++xi) 
                grad[alpha*numPerSubproblem+i*numLabels+xi] += minTable[xi]/expSum;
    }
    for (int i = 0; i < n; ++i)
        grad[i] = -grad[i];
    double objective = -(regularizerObjective + dataObjective + unaryObjective);
    std::cout << "Evaluate: " << objective << "\t(" << regularizerObjective << ", " << dataObjective << ", " << unaryObjective << ")\n";
    return objective;
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
    Array<D> b = 2*Ht(y);
    Array<D> x = b;
    LinearSystem<D> Q = [&](const Array<D>& x) -> Array<D> { return Ht(H(x)) + 0.03*x; };

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
    for (size_t i = 0; i < x.num_elements(); ++i) {
        x.data()[i] = 0;
    }
    quadraticMin<D>(Q, b, x);

    lbfgs_parameter_t params;
    double fVal = 0;
    lbfgs_parameter_init(&params);
    auto algData = DeconvolveData<D>{x, b, Q, R, numLambda};
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
