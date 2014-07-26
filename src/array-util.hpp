#ifndef _ARRAY_UTIL_HPP_
#define _ARRAY_UTIL_HPP_

#include <type_traits>
#include <boost/multi_array.hpp>

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


template <typename A1, typename A2>
void plusEquals(A1& a1, const A2& a2) {
    arrayMap([](double& x1, const double& x2) { x1 += x2; }, a1, a2);
}



#endif
