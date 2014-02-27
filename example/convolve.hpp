#ifndef _EXAMPLE_CONVOLVE_HPP_
#define _EXAMPLE_CONVOLVE_HPP_

#include "deconvolve.hpp"
deconvolution::Array<2> convolve(const deconvolution::Array<2>& im, const deconvolution::Array<2>& ker);
deconvolution::Array<2> convolveFFT(const deconvolution::Array<2>& im, const deconvolution::Array<2>& ker);

#endif
