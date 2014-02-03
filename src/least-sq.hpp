#ifndef _DECONVOLUTION_LEAST_SQ_HPP_
#define _DECONVOLUTION_LEAST_SQ_HPP_

#include "deconvolve.hpp"

namespace deconvolution {
template <int D>
void leastSquares(const Array<D>& y, const LinearSystem<D>& H, Array<D>& x);
}

#endif
