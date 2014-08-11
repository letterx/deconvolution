#ifndef _DECONVOLUTION_UTIL_HPP_
#define _DECONVOLUTION_UTIL_HPP_

#include "deconvolve.hpp"
#include <array>

namespace deconvolution {

template <unsigned long D>
Array<D> elementMult(const Array<D>& a1, const Array<D>& a2) {
    Array<D> result = a1;
    for (size_t i = 0; i < a1.num_elements(); ++i) {
        result.data()[i] *= a2.data()[i];
    }
    return result;
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

template <unsigned long D, unsigned long Level>
struct ArrayForEach {
    typedef std::array<int, D> IdxArray;
    template <typename Fn>
    void forEach(const IdxArray& order, const IdxArray& extents, const IdxArray& strides, int baseIdx, Fn f) {
        const int currDim = order[Level];
        const int stride = strides[currDim];
        for (int i = 0; i < extents[currDim]; ++i) {
            ArrayForEach<D, Level+1> ForEach {};
            ForEach.forEach(order, extents, strides, baseIdx, f);
            baseIdx += stride;
        }
    }
};

template <unsigned long D>
struct ArrayForEach<D, D> {
    typedef std::array<int, D> IdxArray;
    template <typename Fn>
    void forEach(const IdxArray& order, const IdxArray& extents, const IdxArray& strides, int baseIdx, Fn f) {
        f(baseIdx);
    }
};

template <unsigned long D>
using IdxArray = std::array<int, D>;

template <unsigned long D, unsigned long Level, typename Fn>
void arrayForEachTail(const IdxArray<D>& order, const IdxArray<D>& extents, const IdxArray<D>& strides, int baseIdx, Fn f) {
    ArrayForEach<D, Level> ForEach {};
    ForEach.forEach(order, extents, strides, baseIdx, f);
}


template <typename T, unsigned long D>
std::array<T, D> bringToFront(const std::array<T, D>& a, unsigned long idx) {
    std::array<T, D> ret = a;
    auto iter = std::begin(ret);
    *(iter++) = a[idx];
    for (unsigned long i = 0; i < D; ++i) {
        if (i == idx) continue;
        *(iter++) = a[i];
    }
    assert(iter == std::end(ret));
    return ret;
}

template <int D, typename T>
std::array<T, D> arrFromPtr(const T* ptr) {
    std::array<T, D> ret;
    for (int i = 0; i < D; ++i) ret[i] = ptr[i];
    return ret;
}

template <int D, typename T>
std::array<T, D> arrFromVec(const std::vector<T>& vec) {
    std::array<T, D> ret;
    for (int i = 0; i < D; ++i) ret[i] = vec[i];
    return ret;
}

template <unsigned long D> 
IdxArray<D> stridesFromExtents(const IdxArray<D>& extents) {
    IdxArray<D> s;
    s[D-1] = 1;
    for (unsigned long i = 2; i <= D; ++i) {
        s[D-i] = s[D-i+1]*extents[D-i+1];
    }
    return s;
}

template <int D>
int idxFromPt(const IdxArray<D>& point, const IdxArray<D>& strides) {
    int idx = 0;
    for (int i = 0; i < D; ++i) idx += point[i]*strides[i];
    return idx;
}

}

#endif
