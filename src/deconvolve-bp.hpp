#ifndef _DECONVOLVE_BP_HPP_
#define _DECONVOLVE_BP_HPP_

#include <array>
#include "deconvolve.hpp"
#include "regularizer.hpp"
#include "convex-fn.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#include "optimization.h"
#pragma clang diagnostic pop

namespace deconvolution{

using namespace alglib;

template <int D>
boost::general_storage_order<D+1> lambdaOrder(int majorDim);

template <int D, typename Shape>
std::vector<Array<D+1>> allocLambda(const Shape& shape);

template <int D>
void nuUnaries(const Regularizer<D>& R, const Array<D>& nu, Array<D+1>& result);

template <int D>
double dualObjective(const Regularizer<D>& R,
        const LinearSystem<D>& Q,
        const Array<D>& b, 
        const Array<D>& nu,
        double constantTerm,
        const std::vector<Array<D+1>>& lambda);

}

// Implementation
#include "array-util.hpp"

namespace deconvolution {

template <int D>
struct NuOptimizeLBFGS {
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
            static_cast<NuOptimizeLBFGS*>(instance)
                ->_evaluate(lbfgsX, objective, lbfgsGrad);
        }

        static void progress(const real_1d_array& lbfgsX,
                double fx,
                void *instance)
        {
            static_cast<NuOptimizeLBFGS*>(instance)
                ->_progress(lbfgsX, fx);
        }
        static std::vector<ConvexFn> sumLambda(
                const std::vector<Array<D+1>>& lambda,
                const Regularizer<D>& R);

        NuOptimizeLBFGS(const LinearSystem<D>& Q,
                const Array<D>& b,
                const Regularizer<D>& R,
                const std::vector<Array<D+1>>& lambda,
                double t)
            : _Q(Q)
            , _b(b)
            , _R(R)
            , _lambda(lambda) 
            , _lambdaSum(sumLambda(lambda, R))
            , _t(t)
        { }

        void _evaluate(const real_1d_array& lbfgsX,
                double& objective,
                real_1d_array& lbfgsGrad);
        void _progress(const real_1d_array& lbfgsX, double fx);

    protected:

        const LinearSystem<D>& _Q;
        const Array<D>& _b;
        const Regularizer<D>& _R;
        const std::vector<Array<D+1>>& _lambda;
        std::vector<ConvexFn> _lambdaSum;
        const double _t;

};

template <int D>
boost::general_storage_order<D+1> lambdaOrder(int majorDim) {
    std::array<size_t, D+1> order;
    std::array<bool, D+1> ascending;
    for (int i = 0; i < D+1; ++i) {
        order[i] = D - i;
        ascending[i] = true;
    }
    std::swap(order[1], order[D-majorDim]);
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
void nuUnaries(const Regularizer<D>& R, const Array<D>& nu, Array<D+1>& result) {
    int var = 0;
    int label = 0;
    arrayMap(
            [&](double& unary) {
                auto lb = R.getIntervalLB(var, label);
                auto ub = R.getIntervalUB(var, label);
                auto nu_i = nu.data()[var];
                unary = std::min(nu_i*lb, nu_i*ub);
                label++;
                if (label == R.maxLabels()) {
                    label = 0;
                    var++;
                }
            },
            result);
}

template <int D>
Array<D+1> sumUnaries(const Regularizer<D>& R, const Array<D>& nu, const std::vector<Array<D+1>>& lambda) {
    auto lambdaShape = arrayExtents(lambda[0]);
    Array<D+1> unaries(lambdaShape);
    nuUnaries(R, nu, unaries);
    for (int i = 0; i < D; ++i)
        plusEquals(unaries, lambda[i]);
    return unaries;
}

}

#endif
