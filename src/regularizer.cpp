#include "regularizer.hpp"

namespace deconvolution {

template <int D>
double GridRegularizer<D>::evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient) const {
    return 0;
}

#define INSTANTIATE_DECONVOLVE_REGULARIZER(d) \
    template class GridRegularizer<d>;
INSTANTIATE_DECONVOLVE_REGULARIZER(1)
INSTANTIATE_DECONVOLVE_REGULARIZER(2)
INSTANTIATE_DECONVOLVE_REGULARIZER(3)
#undef INSTANTIATE_DECONVOLVE_REGULARIZER

}
