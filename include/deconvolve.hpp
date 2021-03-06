#ifndef _DECONVOLVE_HPP_
#define _DECONVOLVE_HPP_

#include <functional>
#ifdef NDEBUG
#define BOOST_DISABLE_ASSERTS
#endif
#include <boost/multi_array.hpp>

namespace deconvolution {

template <int D>
using Array = boost::multi_array<double, D>;
template <int D>
using LinearSystem = std::function<Array<D>(const Array<D>&)>;

template <int D> class Regularizer;

struct DeconvolveParams {
    double dataSmoothing {0.03};
    double maxSmoothing {1000.0};
    double minSmoothing { 10.0 };
    int maxIterations { 1000 };
    double admmRho { 10.0 };
    double admmConvergenceNorm { 1.0 };
    int lbfgsIters { 20 };
};

struct DeconvolveStats {
    double iterTime = 0;
    double regularizerTime = 0;
    double unaryTime = 0;
    double dataTime = 0;
};

template <int D>
using ProgressCallback = std::function<void(const Array<D>& x, double dual, double primalData, double primalReg, double smoothing)>;
/* 
 * Solve a linear inverse system of the form
 * min_x |y - Hx|_2^2 + R(x)
 * where H is a linear system, R is a regularizer, and y are some given 
 * observables. 
 */
template <int D>
Array<D> Deconvolve(const Array<D>& y, const LinearSystem<D>& H, const LinearSystem<D>& Ht, Regularizer<D>& R, ProgressCallback<D>& pc, DeconvolveParams& params, DeconvolveStats& s);

template <int D>
Array<D> DeconvolveADMM(const Array<D>& y, const LinearSystem<D>& H, const LinearSystem<D>& Ht, Regularizer<D>& R, ProgressCallback<D>& pc, DeconvolveParams& params, DeconvolveStats& s);

template <int D>
Array<D> DeconvolvePrimal(const Array<D>& y, const LinearSystem<D>& H, const LinearSystem<D>& Ht, Regularizer<D>& R, ProgressCallback<D>& pc, DeconvolveParams& params, DeconvolveStats& s, const Array<D>& initX);


} // namespace deconvolution



#endif
