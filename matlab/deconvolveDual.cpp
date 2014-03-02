#include "mex.h"
#include "deconvolve.hpp"
#include "regularizer.hpp"
#include <vector>
#include <exception>

using namespace deconvolution;

class DeconvolutionMexException : public std::exception {
    public:
    DeconvolutionMexException(const char* msg) : _msg(msg) { }
    const char* what() const noexcept { return _msg; }
    const char* _msg;
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 4 || nlhs != 1) {
        mexErrMsgTxt("Usage: x = deconvolveDual(H, Ht, y, @progress)");
    }

    const mxArray* mex_H = prhs[0];
    const mxArray* mex_Ht = prhs[1];
    const mxArray* mex_y = prhs[2];
    const mxArray* mex_progress = prhs[3];
    const int numDims = mxGetNumberOfDimensions(mex_y);
    if (numDims != 3)
        mexErrMsgTxt("Input error: y must be 3-dimensional array");
    const std::vector<int> matlabDims(mxGetDimensions(mex_y), mxGetDimensions(mex_y)+numDims);
    const std::vector<int> dims(matlabDims.rbegin(), matlabDims.rend());
    const int size = std::accumulate(begin(dims), end(dims), 0);

    Array<3> y{dims};
    y.assign(mxGetPr(mex_y), mxGetPr(mex_y)+size);

    LinearSystem<3> H = [&](const Array<3>& x) -> Array<3> {
        mxArray* callbackRhs[] = { const_cast<mxArray*>(mex_H), mxCreateNumericArray(3, matlabDims.data(), mxDOUBLE_CLASS, mxREAL) };
        mxArray* callbackLhs[] = { nullptr };
        double* argData = mxGetPr(callbackRhs[1]);
        for (int i = 0; i < size; ++i) 
            argData[i] = x.data()[i];
        mexCallMATLAB(1, callbackLhs, 2, callbackRhs, "feval");
        Array<3> result{dims};
        double *lhsData = mxGetPr(callbackLhs[0]);
        for (int i = 0; i < size; ++i)
            result.data()[i] = lhsData[i];
        mxDestroyArray(callbackRhs[1]);
        mxDestroyArray(callbackLhs[0]);
        return result;
    };
    LinearSystem<3> Ht = [&](const Array<3>& x) -> Array<3> {
        mxArray* callbackRhs[] = { const_cast<mxArray*>(mex_Ht), mxCreateNumericArray(3, matlabDims.data(), mxDOUBLE_CLASS, mxREAL) };
        mxArray* callbackLhs[] = { nullptr };
        double* argData = mxGetPr(callbackRhs[1]);
        for (int i = 0; i < size; ++i) 
            argData[i] = x.data()[i];
        mexCallMATLAB(1, callbackLhs, 2, callbackRhs, "feval");
        Array<3> result{dims};
        double *lhsData = mxGetPr(callbackLhs[0]);
        for (int i = 0; i < size; ++i)
            result.data()[i] = lhsData[i];
        mxDestroyArray(callbackRhs[1]);
        mxDestroyArray(callbackLhs[0]);
        return result;
    };
    DummyRegularizer<3> R{};
    ProgressCallback<3> pc = [&](const Array<3>& x) { 
        mxArray* callbackRhs[] = { const_cast<mxArray*>(mex_progress), mxCreateDoubleScalar(0.0) };
        mexCallMATLAB(0, nullptr, 2, callbackRhs, "feval");
        mxDestroyArray(callbackRhs[1]);
        throw DeconvolutionMexException{"Terminating in progress callback"};
    };
    DeconvolveStats stats;

    try {
        auto x = Deconvolve<3>(y, H, Ht, R, pc, stats);
    } catch (const DeconvolutionMexException& e) {
        mexErrMsgIdAndTxt("Deconvolve:main", "Exception occurred: %s", e.what());
    } catch (...) {
        mexErrMsgIdAndTxt("Deconvolve:unknown", "Unknown Exception occurred!");
    }


    plhs[0] = mxCreateNumericArray(3, matlabDims.data(), mxDOUBLE_CLASS, mxREAL);

    mexPrintf("Success!\n");
}
