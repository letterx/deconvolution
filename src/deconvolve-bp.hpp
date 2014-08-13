#ifndef _DECONVOLVE_BP_HPP_
#define _DECONVOLVE_BP_HPP_

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
void addUnaries(const Regularizer<D>& R, const Array<D>& nu, Array<D+1>& result);
template <int D>
double dualObjective(const Regularizer<D>& R,
        const LinearSystem<D>& Q,
        const Array<D>& b, 
        const Array<D>& nu,
        double constantTerm,
        const std::vector<Array<D+1>>& lambda);

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
            , _lambdaSum(sumLambda(lambda, R))
            , _t(t)
        { }

        void _evaluate(const real_1d_array& lbfgsX,
                double& objective,
                real_1d_array& lbfgsGrad);
        void _progress(const real_1d_array& lbfgsX, double fx);
        std::vector<ConvexFn> sumLambda(const std::vector<Array<D+1>>& lambda,
                const Regularizer<D>& R) const;

        const LinearSystem<D>& _Q;
        const Array<D>& _b;
        const Regularizer<D>& _R;
        const std::vector<Array<D+1>>& _lambda;
        std::vector<ConvexFn> _lambdaSum;
        const double _t;

};


}

#endif
