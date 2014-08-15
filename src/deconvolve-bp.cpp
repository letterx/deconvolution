#include "deconvolve-bp.hpp"

#include <limits>
#include <iostream>
#include <chrono>

#include "quadratic-min.hpp"
#include "util.hpp"
#include "regularizer.hpp"
#include "array-util.hpp"
#include "convex-fn.hpp"


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#include "optimization.h"
#pragma clang diagnostic pop

namespace deconvolution{

using namespace alglib;

template <int D>
double dualObjective(const Regularizer<D>& R,
        const LinearSystem<D>& Q,
        const Array<D>& b, 
        const Array<D>& nu,
        double constantTerm,
        const std::vector<Array<D+1>>& lambda) {
    Array<D> bPlusNu = b;
    plusEquals(bPlusNu, nu);
    Array<D> x = bPlusNu;
    double dataObjective = quadraticMinCG<D>(Q, bPlusNu, x) + constantTerm; 

    double unaryObjective = 0;
    auto unaries = sumUnaries(R, nu, lambda);
    arraySubMap<1>(
            [&] (const typename Array<D+1>::template subarray<1>::type& unary_i) {
                unaryObjective += arrayMin(unary_i);
            },
            unaries);

    double regularizerObjective = 0;
    for (int i = 0; i < D; ++i)
        regularizerObjective += R.minMarginal(i, -1.0*lambda[i], unaries);

    return dataObjective + unaryObjective + regularizerObjective;
}

template <int D>
void nuOptimizeLBFGS<D>::optimize(const LinearSystem<D>& Q,
        const Array<D>& b,
        const Regularizer<D>& R,
        const std::vector<Array<D+1>>& lambda,
        Array<D>& nu) {

    real_1d_array lbfgsX;
    lbfgsX.setcontent(nu.num_elements(), nu.data());

    minlbfgsstate lbfgsState;
    minlbfgsreport lbfgsReport;

    minlbfgscreate(10, lbfgsX, lbfgsState);
    minlbfgssetxrep(lbfgsState, true);

    const double t = 0.001;
    auto algData = nuOptimizeLBFGS<D>{Q, b, R, lambda, t};
    minlbfgssetcond(lbfgsState, 0.0, 0.0, 0, 10);

    minlbfgsoptimize(lbfgsState, 
            nuOptimizeLBFGS<D>::evaluate, 
            nuOptimizeLBFGS<D>::progress, 
            &algData);
    minlbfgsresults(lbfgsState, lbfgsX, lbfgsReport);

    auto shape = arrayExtents(b);
    Array<D> x{shape};
    for (size_t i = 0; i < x.num_elements(); ++i)
        x.data()[i] = lbfgsX.getcontent()[i];
    nu = -2.0 * Q(x) + b;
}

template <int D>
void nuOptimizeLBFGS<D>::_evaluate(const real_1d_array& lbfgsX,
                double& objective,
                real_1d_array& lbfgsGrad) {
    auto shape = arrayExtents(_b);
    auto x = ConstArrayRef<D>{lbfgsX.getcontent(), shape};
    const int n = x.num_elements();
    auto Qx = _Q(x);
    auto xQx = dot(x, Qx);
    auto bx = dot(_b, x);
    for (int i = 0; i < n; ++i)
        lbfgsGrad.getcontent()[i] = 2*Qx.data()[i] - _b.data()[i];

    objective = xQx - bx;

    assert(n == static_cast<int>(_lambdaSum.size()));
    for (int i = 0; i < n; ++i) {
        objective += _lambdaSum[i].moreauEnvelope(x.data()[i], _t);
        lbfgsGrad.getcontent()[i] -= _lambdaSum[i].moreauGrad(x.data()[i], _t);
    }


}

template <int D>
void nuOptimizeLBFGS<D>::_progress(const real_1d_array& lbfgsX, double fx) {

}

template <int D>
std::vector<ConvexFn> nuOptimizeLBFGS<D>::sumLambda(
        const std::vector<Array<D+1>>& lambda,
        const Regularizer<D>& R) const {
    // typedef for array of lambda for a single variable
    typedef typename Array<D+1>::template subarray<1>::type Lambda_i;

    Array<D+1> lambdaSum{arrayExtents(lambda[0])};
    for (const auto& l : lambda)
        plusEquals(lambdaSum, l);

    std::vector<ConvexFn> convexLambda;
    int i = 0;
    arraySubMap<1>(
            [&] (const Lambda_i& lambda_i) {
                const int numL = R.numLabels(i);
                DECONV_ASSERT(numL <= static_cast<int>(lambda_i.size()));
                std::vector<double> xVals;
                std::vector<double> fVals;
                xVals.push_back(R.getIntervalLB(i, 0));
                fVals.push_back(lambda_i[0]);
                for (int l = 0; l < numL-1; ++l) {
                    xVals.push_back(R.getIntervalUB(i, l));
                    fVals.push_back(std::min(lambda_i[l], lambda_i[l+1]));
                }
                xVals.push_back(R.getIntervalUB(i, numL-1));
                fVals.push_back(lambda_i[numL-1]);
                assert(static_cast<int>(xVals.size()) == numL+1 && 
                    static_cast<int>(fVals.size()) == numL+1);
                auto fn = PiecewiseLinearFn{xVals.begin(), xVals.end(), fVals.begin()};
                convexLambda.emplace_back(fn.convexify());
                i++;
            },
            lambdaSum);
    return convexLambda;
}

template <int D>
Array<D> DeconvolveConvexBP(
        const Array<D>& y,
        const LinearSystem<D>& H,
        const LinearSystem<D>& Ht,
        Regularizer<D>& R,
        ProgressCallback<D>& pc,
        DeconvolveParams& params,
        DeconvolveStats& s) {
    Array<D> b = 2*Ht(y);
    const auto shape = arrayExtents(b);
    Array<D> x(shape);
    LinearSystem<D> Q = [&](const Array<D>& x) -> Array<D> { 
        return Ht(H(x)) + params.dataSmoothing*x; 
    };
    double constantTerm = dot(y, y);

    int numPrimalVars = x.num_elements();
    const auto lambdaShape = arrayExtents(x)[R.maxLabels()];
    auto lambda = allocLambda<D>(lambdaShape);
    auto primalMu_i = std::vector<double>(numPrimalVars*R.maxLabels(), 0.0);

    auto modifiedUnaries = Array<D+1>{lambdaShape};
    auto nu = Array<D>{shape};

    // Initialize x to 0 to find the least norm solution to Qx = b
    for (size_t i = 0; i < x.num_elements(); ++i)
        x.data()[i] = 0;

    std::cout << "Finding least-squares fit\n";
    quadraticMinCG<D>(Q, b, x);
    auto dualObj = dualObjective(R, Q, b, nu, constantTerm, lambda);

    bool converged = false;
    int iter = 0;
    while (!converged) {
        std::cout << "Iter: " << iter << "\n";

       // Compute modifiedUnaries = unaries + sum of lambda - lambda_alpha
       nuUnaries(R, nu, modifiedUnaries);
       for (int i = 0; i < D; ++i)
           plusEquals(modifiedUnaries, lambda[i]);
       
       for (int i = 0; i < D; ++i) {
           arrayMap([](double& l, double& u) { u -= l; l = u; }, 
                   lambda[i], modifiedUnaries);
           R.minMarginal(i, lambda[i], lambda[i]);
           arrayMap(
                   [&](double& l, double& u) { 
                       l = l/numPrimalVars - u;
                       u += l;
                   },
                   lambda[i], modifiedUnaries);
           auto newObj = dualObjective(R, Q, b, nu, constantTerm, lambda);
           assert(newObj >= dualObj);
           dualObj = newObj;
           std::cout << "Dual: " << dualObj << "\n";
       }
       // run steps of gradient descent on data-term + soft-min of modified unaries
       
       nuOptimizeLBFGS<D>::optimize(Q, b, R, lambda, nu);

       auto newObj = dualObjective(R, Q, b, nu, constantTerm, lambda);
       assert(newObj >= dualObj);
       dualObj = newObj;

       std::cout << "Dual: " << dualObj << "\n";

       iter++;

    }

    return x;
}

#define INSTANTIATE_DECONVOLVE(d)                                              \
    template Array<d> DeconvolveConvexBP<d>(const Array<d>& y,                 \
		const LinearSystem<d>& H,                                              \
		const LinearSystem<d>& Q,                                              \
		Regularizer<d>& R,                                                     \
		ProgressCallback<d>& pc,                                               \
		DeconvolveParams& params,                                              \
		DeconvolveStats& s);                                                   \
    template boost::general_storage_order<d+1> lambdaOrder<d>(int majorDim);   

INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
