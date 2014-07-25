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




#endif
