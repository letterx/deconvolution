#ifndef _DECONVOLUTION_OPTIMAL_GRAD_HPP_
#define _DECONVOLUTION_OPTIMAL_GRAD_HPP_

#include <lbfgs.h>

namespace deconvolution {

int optimalGradDescent(int n, double *x, double *fval, lbfgs_evaluate_t eval, lbfgs_progress_t prog, void *instance);

}

#endif
