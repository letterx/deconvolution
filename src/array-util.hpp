#ifndef _ARRAY_UTIL_HPP_
#define _ARRAY_UTIL_HPP_

#include <type_traits>
#include <boost/multi_array.hpp>

namespace deconvolution {

template <typename A, size_t D>
typename std::enable_if<D == 0, boost::multi_array_types::extent_gen>::type
arrayExtentsHelper(const A& array) {
    return boost::extents;
}

template <typename A, size_t D>
typename std::enable_if<(D > 0), 
         typename boost::multi_array_types::extent_gen::gen_type<D>::type >::type
arrayExtentsHelper(const A& array) 
{
    return arrayExtentsHelper<A, D-1>(array)
        [boost::multi_array_types::extent_range(array.index_bases()[D-1], 
            array.index_bases()[D-1] + array.shape()[D-1])];
}

template <typename A>
auto
arrayExtents(const A& array)
    -> decltype(arrayExtentsHelper<A, A::dimensionality>(array))
{
    return arrayExtentsHelper<A, A::dimensionality>(array);
}



/*
 * arrayMap
 */

template <typename A>
struct ArrayDim {
    static const size_t value = std::remove_reference<A>::type::dimensionality;
};

template <typename Fn, typename A1>
typename std::enable_if<ArrayDim<A1>::value == 1, void>::type
arrayMap(Fn f, A1&& a1) {
    auto i1 = a1.begin();
    auto e1 = a1.end();
    for (; i1 != e1; ++i1)
        f(*i1);
}


template <typename Fn, typename A1>
typename std::enable_if<(ArrayDim<A1>::value > 1), void>::type
arrayMap(Fn f, A1&& a1) {
    auto i1 = a1.begin();
    auto e1 = a1.end();
    for (; i1 != e1; ++i1)
        arrayMap(f, *i1);
}


template <typename Fn, typename A1, typename A2>
typename std::enable_if<ArrayDim<A1>::value == 1, void>::type
arrayMap(Fn f, A1&& a1, A2&& a2) {
    static_assert(ArrayDim<A1>::value == ArrayDim<A2>::value, 
            "Array dims must match");
    auto i1 = a1.begin();
    auto i2 = a2.begin();
    auto e1 = a1.end();
    for (; i1 != e1; ++i1, ++i2)
        f(*i1, *i2);
}


template <typename Fn, typename A1, typename A2>
typename std::enable_if<(ArrayDim<A1>::value > 1), void>::type
arrayMap(Fn f, A1&& a1, A2&& a2) {
    static_assert(ArrayDim<A1>::value == ArrayDim<A2>::value, 
            "Array dims must match");
    auto i1 = a1.begin();
    auto i2 = a2.begin();
    auto e1 = a1.end();
    for (; i1 != e1; ++i1, ++i2)
        arrayMap(f, *i1, *i2);
}

template <int I, typename Fn, typename A>
typename std::enable_if<(ArrayDim<A>::value == I), void>::type
arraySubMap(Fn&& f, A&& a) {
    f(a);
}

template <int I, typename Fn, typename A>
typename std::enable_if<(ArrayDim<A>::value > I), void>::type
arraySubMap(Fn&& f, A&& a) {
    auto iter = a.begin();
    auto e = a.end();
    for (; iter != e; ++iter)
        arraySubMap<I>(f, *iter);
}

template <typename A1, typename A2>
void plusEquals(A1& a1, const A2& a2) {
    arrayMap([](double& x1, const double& x2) { x1 += x2; }, a1, a2);
}

template <typename A1, typename A2>
void minusEquals(A1& a1, const A2& a2) {
    arrayMap([](double& x1, const double& x2) { x1 -= x2; }, a1, a2);
}

template <typename A>
typename A::element arrayMin(const A& a) {
    typename A::element minVal = std::numeric_limits<typename A::element>::max();
    arrayMap([&](typename A::element val) { minVal = std::min(minVal, val); }, a);
    return minVal;
}

template <typename A1, typename A2>
typename A1::element dot(const A1& a1, const A2& a2) {
    typename A1::element sum = 0;
    arrayMap([&](typename A1::element v1,
                typename A2::element v2) { sum += v1*v2; }, a1, a2);
    return sum;
}

template <typename A>
typename A::element norm(const A& a) {
    return dot(a,a);
}


}
#endif
