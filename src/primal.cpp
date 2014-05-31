#include "deconvolve.hpp"

#include <limits>
#include <iostream>

#include "util.hpp"
#include "regularizer.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#include "optimization.h"
#pragma clang diagnostic pop

namespace deconvolution {

using namespace alglib;

template <int D>
struct PrimalLbfgsData {
    public:
        void evaluate(const real_1d_array& lbfgsX,
                double& obj,
                real_1d_array& lbfgsGrad);

        void progress(const real_1d_array& lbfgsX,
                double fx);

        LinearSystem<D>& _Q;
        Array<D>& _b;
        Array<D>& _x;
        Regularizer<D>& _R;
        double _constantTerm;
        ProgressCallback<D>& _pc;
};

template <int D>
void PrimalLbfgsData<D>::evaluate(const real_1d_array& lbfgsX,
                double& obj,
                real_1d_array& lbfgsGrad) {
    int n = _x.num_elements();
    for (int i = 0; i < n; ++i) {
        _x.data()[i] = lbfgsX.getcontent()[i];
    }
    auto Qx = _Q(_x);
    auto xQx = dot(_x, Qx);
    auto bx = dot(_b, _x);
    for (int i = 0; i < n; ++i)
        lbfgsGrad[i] = 2*Qx.data()[i] - _b.data()[i];

    obj = xQx - bx;

    obj += _R.primal(_x.data(), lbfgsGrad.getcontent());
}

template <int D>
void PrimalLbfgsData<D>::progress(const real_1d_array& lbfgsX,
                double fx) {
    std::cout << "Iteration done: " << fx << "\n";
    _pc(_x, fx, 0, 0, 0);
}


template <int D>
static void lbfgsEvaluate(
        const real_1d_array& lbfgsX,
        double& objective,
        real_1d_array& lbfgsGrad,
        void* instance) {
    static_cast<PrimalLbfgsData<D>*>(instance)
        ->evaluate(lbfgsX, objective, lbfgsGrad);
}

template <int D>
static void lbfgsProgress(
        const real_1d_array& lbfgsX,
        double fx,
        void *instance) {
    static_cast<PrimalLbfgsData<D>*>(instance)->progress(lbfgsX, fx);
}

template <int D>
Array<D> DeconvolvePrimal(const Array<D>& y,
        const LinearSystem<D>& H,
        const LinearSystem<D>& Ht,
        Regularizer<D>& R,
        ProgressCallback<D>& pc,
        DeconvolveParams& params,
        DeconvolveStats& s) {
    Array<D> b = 2*Ht(y);
    Array<D> x = b;
    LinearSystem<D> Q = [&](const Array<D>& x) -> Array<D> { return Ht(H(x)); };
    double constantTerm = dot(y, y);

    int numPrimalVars = x.num_elements();
    for (int i = 0; i < numPrimalVars; ++i) {
        x.data()[i] = 0;
    }

    real_1d_array lbfgsX;
    lbfgsX.setcontent(numPrimalVars, x.data());

    minlbfgsstate lbfgsState;
    minlbfgsreport lbfgsReport;

    minlbfgscreate(10, lbfgsX, lbfgsState);
    minlbfgssetxrep(lbfgsState, true);
    minlbfgssetcond(lbfgsState, 1e-5, 0.0, 0, 1000);

    auto algData = PrimalLbfgsData<D>{Q, b, x, R, constantTerm, pc};

    std::cout << "Begin lbfgs\n";
    minlbfgsoptimize(lbfgsState, lbfgsEvaluate<D>, lbfgsProgress<D>, &algData);
    minlbfgsresults(lbfgsState, lbfgsX, lbfgsReport);

    for (int i = 0; i < numPrimalVars; ++i) {
        x.data()[i] = lbfgsX.getcontent()[i];
    }

    return x;
}

#define INSTANTIATE_DECONVOLVE(d) \
    template Array<d> DeconvolvePrimal<d>(const Array<d>& y, const LinearSystem<d>& H, const LinearSystem<d>& Q, Regularizer<d>& R, ProgressCallback<d>& pc, DeconvolveParams& params, DeconvolveStats& s); \

INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE

} // namespace deconvolution
