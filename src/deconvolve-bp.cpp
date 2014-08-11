#include "deconvolve.hpp"
#include <limits>
#include <iostream>
#include <chrono>
#include <random>

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
boost::general_storage_order<D+1> lambdaOrder(int majorDim) {
    std::array<size_t, D+1> order;
    std::array<bool, D+1> ascending;
    for (int i = 0; i < D+1; ++i) {
        order[i] = i;
        ascending[i] = true;
    }
    std::swap(order[majorDim], order[D-1]);
    return boost::general_storage_order<D+1>(order.begin(), ascending.begin());
}

template <int D, typename Shape>
std::vector<Array<D+1>> allocLambda(const Shape& shape) {
    std::vector<Array<D+1>> lambda;
    for (int i = 0; i < D; ++i)
        lambda.emplace_back(shape, lambdaOrder<D>(i));
    return lambda;
}

template <int D>
void addUnaries(const Regularizer<D>& R, const Array<D>& nu, Array<D+1>& result) {
    int var = 0;
    int label = 0;
    arrayMap(
            [&](double& unary) {
                auto lb = R.getIntervalLB(var, label);
                auto ub = R.getIntervalUB(var, label);
                auto nu_i = nu.data()[var];
                unary = std::min(nu_i*lb, nu_i*ub);
                label++;
                if (label == R.numLabels()) {
                    label = 0;
                    var++;
                }
            },
            result);
}

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
    auto lambdaShape = arrayExtents(lambda[0]);
    Array<D+1> unaries(lambdaShape);
    addUnaries(R, nu, unaries);
    for (int i = 0; i < D; ++i)
        plusEquals(unaries, lambda[i]);
    arraySubMap<1>(
            [&] (const typename Array<D+1>::template subarray<1>::type& unary_i) {
                unaryObjective += arrayMin(unary_i);
            },
            unaries);

    double regularizerObjective = 0;
    for (int i = 0; i < D; ++i)
        regularizerObjective += R.minMarginal(i, lambda[i], unaries);

    return dataObjective + unaryObjective + regularizerObjective;
}

template <int D>
struct nuOptimizeLBFGS {
    public:
        static void optimize(const LinearSystem<D>& Q,
                const Array<D>& b,
                const Regularizer<D>& R,
                const std::vector<Array<D+1>>& lambda,
                Array<D>& nu);

        static void evaluate(const real_1d_array& lbfgsX,
                double& objective,
                real_1d_array& lbfgsGrad,
                void* instance) 
        {
            static_cast<nuOptimizeLBFGS*>(instance)
                ->_evaluate(lbfgsX, objective, lbfgsGrad);
        }

        static void progress(const real_1d_array& lbfgsX,
                double fx,
                void *instance)
        {
            static_cast<nuOptimizeLBFGS*>(instance)
                ->_progress(lbfgsX, fx);
        }

    protected:
        nuOptimizeLBFGS(const LinearSystem<D>& Q,
                const Array<D>& b,
                const Regularizer<D>& R,
                const std::vector<Array<D+1>>& lambda,
                double t)
            : _Q(Q)
            , _b(b)
            , _R(R)
            , _lambda(lambda) 
            , _t(t)
        { }

        void _evaluate(const real_1d_array& lbfgsX,
                double& objective,
                real_1d_array& lbfgsGrad);
        void _progress(const real_1d_array& lbfgsX, double fx);

        const LinearSystem<D>& _Q;
        const Array<D>& _b;
        const Regularizer<D>& _R;
        const std::vector<Array<D+1>>& _lambda;
        const double _t;

        std::vector<ConvexFn> _lambdaSum;
};

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
    const auto lambdaShape = arrayExtents(x)[R.numLabels()];
    auto lambda = allocLambda<D>(lambdaShape);
    auto primalMu_i = std::vector<double>(numPrimalVars*R.numLabels(), 0.0);

    auto modifiedUnaries = Array<D+1>{lambdaShape};
    auto nu = Array<D>{shape};

    // Initialize x to 0 to find the least norm solution to Qx = b
    for (size_t i = 0; i < x.num_elements(); ++i)
        x.data()[i] = 0;

    std::cout << "Finding least-squares fit\n";
    quadraticMinCG<D>(Q, b, x);

    bool converged = false;
    while (!converged) {

       // Compute modifiedUnaries = unaries + sum of lambda - lambda_alpha
       addUnaries(R, nu, modifiedUnaries);
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
       }
       // run steps of gradient descent on data-term + soft-min of modified unaries
       
       nuOptimizeLBFGS<D>::optimize(Q, b, R, lambda, nu);

       auto dualObj = dualObjective(R, Q, b, nu, constantTerm, lambda);
       std::cout << "Dual: " << dualObj << "\n";
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
		DeconvolveStats& s);

INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
