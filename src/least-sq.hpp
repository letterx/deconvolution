#ifndef _DECONVOLUTION_LEAST_SQ_HPP_
#define _DECONVOLUTION_LEAST_SQ_HPP_

#include "deconvolve.hpp"

namespace deconvolution {
template <int D>
void leastSquares(const LinearSystem<D>& Q, const Array<D>& Ht_y, Array<D>& x);
}

#endif
