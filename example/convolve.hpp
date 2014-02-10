#ifndef _EXAMPLE_CONVOLVE_HPP_
#define _EXAMPLE_CONVOLVE_HPP_

#include "deconvolve.hpp"
#include <vector>
#include <iostream>

inline deconvolution::Array<2> convolve(const deconvolution::Array<2>& im, const deconvolution::Array<2>& ker) {
    // Make sure that (0,0) contained in kernel, otherwise things get wonky
    assert(ker.index_bases()[0] <= 0 && 0 < ker.index_bases()[0] + ker.shape()[0]);
    assert(ker.index_bases()[1] <= 0 && 0 < ker.index_bases()[1] + ker.shape()[1]);

    typedef deconvolution::Array<2> A;
    typedef A::index index;
    typedef A::size_type size_type;
    std::vector<index> bases = { 
        index(im.index_bases()[0] - (ker.index_bases()[0] + ker.shape()[0] - 1)),
        index(im.index_bases()[1] - (ker.index_bases()[1] + ker.shape()[1] - 1)) };
    std::vector<size_type> shape = { 
        im.shape()[0] + ker.shape()[0] - 1,
        im.shape()[1] + ker.shape()[1] - 1 };
    A pad;
    pad.resize(shape);
    pad.reindex(bases);

    for (int i = im.index_bases()[0]; 
            i != im.index_bases()[0] + int(im.shape()[0]);
            ++i) {
        for (int j = im.index_bases()[1];
                j != im.index_bases()[1] + int(im.shape()[1]);
                ++j) {
            pad[i][j] = im[i][j];
        }
    }

    A result = im;
    assert(result.index_bases()[0] == im.index_bases()[0]
            && result.index_bases()[1] == im.index_bases()[1]);
    assert(result.shape()[0] == im.shape()[0] 
            && result.shape()[1] == im.shape()[1]);
    for (int i = result.index_bases()[0];
            i != result.index_bases()[0] + int(result.shape()[0]);
            ++i) {
        for (int j = result.index_bases()[1];
                j != result.index_bases()[1] + int(result.shape()[1]);
                ++j) {
            result[i][j] = 0;
            for (int k = ker.index_bases()[0];
                    k != ker.index_bases()[0] + int(ker.shape()[0]);
                    ++k) {
                for (int l = ker.index_bases()[1];
                        l != ker.index_bases()[1] + int(ker.shape()[1]);
                        ++l) {
                    result[i][j] += ker[k][l] * pad[i-k][j-l];
                }
            }
        }
    }

    return result;
}

#endif
