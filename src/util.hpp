#ifndef _DECONVOLUTION_UTIL_HPP_
#define _DECONVOLUTION_UTIL_HPP_

#include "deconvolve.hpp"

namespace deconvolution {
template <unsigned long D>
double dot(const Array<D>& a1, const Array<D>& a2) {
    double sum = 0.0;
    for (size_t i = 0; i < a1.num_elements(); ++i) {
        sum += a1.data()[i] * a2.data()[i];
    }
    return sum;
}

template <unsigned long D>
double norm(const Array<D>& a) {
    return dot(a,a);
}

template <unsigned long D>
Array<D> operator*(double d, const boost::multi_array<double, D>& a) {
    Array<D> r = a;
    for (size_t i = 0; i < r.num_elements(); ++i)
        r.data()[i] *= d;
    return r;
}

template <unsigned long D>
Array<D>& operator+=(Array<D>& a1, const Array<D>& a2) {
    for (size_t i = 0; i < a1.num_elements(); ++i)
        a1.data()[i] += a2.data()[i];
    return a1;
}

template <unsigned long D>
Array<D> operator+(const Array<D>& a1, const Array<D>& a2) {
    Array<D> r = a1;
    r += a2;
    return r;
}

template <unsigned long D>
Array<D>& operator-=(Array<D>& a1, const Array<D>& a2) {
    for (size_t i = 0; i < a1.num_elements(); ++i)
        a1.data()[i] -= a2.data()[i];
    return a1;
}

template <unsigned long D>
Array<D> operator-(const Array<D>& a1, const Array<D>& a2) {
    Array<D> r = a1;
    r -= a2;
    return r;
}
}

#endif
