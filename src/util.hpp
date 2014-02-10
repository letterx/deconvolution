#ifndef _DECONVOLUTION_UTIL_HPP_
#define _DECONVOLUTION_UTIL_HPP_

#include "deconvolve.hpp"

template <typename T>
double dot(const T& a1, const T& a2) {
    double sum = 0.0;
    for (size_t i = 0; i < a1.num_elements(); ++i) {
        sum += a1.data()[i] * a2.data()[i];
    }
    return sum;
}

template <unsigned long D>
deconvolution::Array<D> operator*(double d, const boost::multi_array<double, D>& a) {
    deconvolution::Array<D> r = a;
    for (size_t i = 0; i < r.num_elements(); ++i)
        r.data()[i] *= d;
    return r;
}

template <unsigned long D>
deconvolution::Array<D> operator+(const deconvolution::Array<D>& a1, const deconvolution::Array<D>& a2) {
    deconvolution::Array<D> r = a1;
    for (size_t i = 0; i < r.num_elements(); ++i)
        r.data()[i] += a2.data()[i];
    return r;
}

#endif
