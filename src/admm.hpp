#ifndef _ADMM_HPP_
#define _ADMM_HPP_

#include "deconvolve.hpp"
#include "regularizer.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#include "optimization.h"
#pragma clang diagnostic pop

namespace deconvolution {

using namespace alglib;

template <int D>
class AdmmRegularizerLbfgs {
    public:
        AdmmRegularizerLbfgs(int n, Regularizer<D>* R, DeconvolveParams* params,
                real_1d_array* hessianDiagArray, minlbfgsstate* lbfgsState,
                const double* mu1, const double* nu)
            : _numXVars(n)
            , _R(R)
            , _params(params)
            , _hessianDiagArray(hessianDiagArray)
            , _lbfgsState(lbfgsState)
            , _mu1(mu1)
            , _nu(nu)
            { }

        static void evaluate(
            const real_1d_array& lbfgsX,
            double& objective,
            real_1d_array& lbfgsGrad,
            void* instance) 
        {
            auto data = static_cast<AdmmRegularizerLbfgs*>(instance);
            double* hessianDiag = data->_hessianDiagArray->getcontent();
            objective = data->_evaluate(
                    lbfgsX.getcontent(), 
                    lbfgsGrad.getcontent(),
                    hessianDiag);
            for (int i = 0; i < data->_hessianDiagArray->length(); ++i) {
                (*data->_hessianDiagArray)[i] = std::max(hessianDiag[i], 1e-7);
            }
            //FIXME
            //minlbfgssetprecdiag(*data->_lbfgsState, *data->_hessianDiagArray);
            for (int i = 0; i < lbfgsGrad.length(); ++i)
                lbfgsGrad[i] = -lbfgsGrad[i];
            objective = -objective;
        }

        static void progress(
            const real_1d_array& lbfgsX,
            double fx,
            void *instance) 
        {
            static_cast<AdmmRegularizerLbfgs*>(instance)->_progress(
                    lbfgsX.getcontent(), fx);
        }

    public:
        double _evaluate(const double* lambda, 
                double* grad,
                double* hessianDiag);
        double _evaluateUnary(int i, 
                const double* lambda,
                double* grad,
                double* hessianDiag);
        void _progress(const double* lambda, double fx);

        int _numXVars;
        Regularizer<D>* _R;
        DeconvolveParams* _params;
        real_1d_array* _hessianDiagArray;
        minlbfgsstate* _lbfgsState;
        const double* _mu1;
        const double* _nu;
        int _iter = 0;
};

}

#endif
