#include "convolve.hpp"

#include <assert.h>
#include <vector>
#include <iostream>
#include <fftw3.h>

deconvolution::Array<2> convolve(const deconvolution::Array<2>& im, const deconvolution::Array<2>& ker) {
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

double* fftwIn1 = 0;
double* fftwIn2 = 0;
fftw_complex* fftwOut1 = 0;
fftw_complex* fftwOut2 = 0;
fftw_plan fftwForward1;
fftw_plan fftwForward2;
fftw_plan fftwInverse;
std::vector<int> fftwShape;

deconvolution::Array<2> convolveFFT(const deconvolution::Array<2>& im, const deconvolution::Array<2>& ker) {
    // Make sure that (0,0) contained in kernel, otherwise things get wonky
    assert(ker.index_bases()[0] <= 0 && 0 < ker.index_bases()[0] + ker.shape()[0]);
    assert(ker.index_bases()[1] <= 0 && 0 < ker.index_bases()[1] + ker.shape()[1]);

    typedef deconvolution::Array<2> A;
    typedef A::index index;
    typedef A::size_type size_type;
    std::vector<index> padBases = { im.index_bases()[0], im.index_bases()[1] };
    std::vector<size_type> shape = { 
        im.shape()[0] + ker.shape()[0],
        im.shape()[1] + ker.shape()[1] };
    std::vector<index> kerBases = { ker.index_bases()[0], ker.index_bases()[1] };
    int realN = shape[0]*shape[1];
    int complexN = shape[0]*(shape[1]/2+1);
    if (fftwIn1 == 0) {
        fftwIn1 = fftw_alloc_real(realN);
        fftwIn2 = fftw_alloc_real(realN);
        fftwOut1 = fftw_alloc_complex(complexN);
        fftwOut2 = fftw_alloc_complex(complexN);

        fftwForward1 = fftw_plan_dft_r2c_2d(shape[0], shape[1], fftwIn1, fftwOut1, FFTW_ESTIMATE);
        fftwForward2 = fftw_plan_dft_r2c_2d(shape[0], shape[1], fftwIn2, fftwOut2, FFTW_ESTIMATE);
        fftwInverse = fftw_plan_dft_c2r_2d(shape[0], shape[1], fftwOut1, fftwIn1, FFTW_ESTIMATE);

        fftwShape = std::vector<int>{shape[0], shape[1]};
    } else {
        assert(fftwShape[0] == int(shape[0]) && fftwShape[1] == int(shape[1]));
    }


    boost::multi_array_ref<double, 2> imPad(fftwIn1, shape);
    imPad.reindex(padBases);
    boost::multi_array_ref<double, 2> kerPad(fftwIn2, shape);
    kerPad.reindex(kerBases);
    for (int i = 0; i < realN; ++i)
        fftwIn1[i] = fftwIn2[i] = 0.0;

    for (int i = im.index_bases()[0]; 
            i != im.index_bases()[0] + int(im.shape()[0]);
            ++i)
        for (int j = im.index_bases()[1];
                j != im.index_bases()[1] + int(im.shape()[1]);
                ++j)
            imPad[i][j] = im[i][j];

    for (int i = ker.index_bases()[0];
            i != ker.index_bases()[0] + int(ker.shape()[0]);
            ++i)
        for (int j = ker.index_bases()[1];
                j != ker.index_bases()[1] + int(ker.shape()[1]);
                ++j)
            kerPad[i][j] = ker[i][j];

    fftw_execute(fftwForward1);
    fftw_execute(fftwForward2);

    double scale = 1.0/realN;
    for (int i = 0; i < complexN; ++i) {
        fftw_complex& c1 = fftwOut1[i];
        const fftw_complex& c2 = fftwOut2[i];
        fftw_complex tmp = { 
            c1[0]*c2[0] - c1[1]*c2[1],
            c1[0]*c2[1] + c1[1]*c2[0]
        };
        c1[0] = tmp[0]*scale;
        c1[1] = tmp[1]*scale;
    }

    fftw_execute(fftwInverse);

    std::vector<int> imBase(im.index_bases(), im.index_bases()+2);
    std::vector<int> imShape(im.shape(), im.shape()+2);
    A result(imShape);
    result.reindex(imBase);
    assert(result.index_bases()[0] == im.index_bases()[0]
            && result.index_bases()[1] == im.index_bases()[1]);
    assert(result.shape()[0] == im.shape()[0] 
            && result.shape()[1] == im.shape()[1]);
    padBases[0] += ker.index_bases()[0];
    padBases[1] += ker.index_bases()[1];
    imPad.reindex(padBases);

    for (int i = im.index_bases()[0]; 
            i != im.index_bases()[0] + int(im.shape()[0]);
            ++i)
        for (int j = im.index_bases()[1];
                j != im.index_bases()[1] + int(im.shape()[1]);
                ++j)
            result[i][j] = imPad[i][j];


    return result;

}
