#ifndef _DECONVOLUTION_LEAST_SQ_HPP_
#define _DECONVOLUTION_LEAST_SQ_HPP_

#include "deconvolve.hpp"

namespace deconvolution {
/*
 * quadraticMin: computes min_x x^T Q x - b^T x 
 * returns: value of min
 * Writes minimizing x to last parameter.
 * If input x is close to true value, will speed up computation.
 *
 * Note: x is also the solution to 2Qx = b
 */
template <int D>
double quadraticMin(const LinearSystem<D>& Q, const Array<D>& b, Array<D>& x);
}

#endif
