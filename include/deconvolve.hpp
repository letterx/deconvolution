#ifndef _DECONVOLVE_HPP_
#define _DECONVOLVE_HPP_

#include <functional>
#include <boost/multi_array.hpp>

namespace deconvolution {

template <int D>
using Array = boost::multi_array<double, D>;
template <int D>
using LinearSystem = std::function<Array<D>(const Array<D>&)>;

template <int D> class Regularizer;

/* 
 * Solve a linear inverse system of the form
 * min_x |y - Hx|_2^2 + R(x)
 * where H is a linear system, R is a regularizer, and y are some given 
 * observables. 
 */
template <int D>
Array<D> Deconvolve(const Array<D>& y, const LinearSystem<D>& H, const LinearSystem<D>& Ht, const Regularizer<D>& R);

}

#endif
