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
        mexErrMsgTxt("Usage: x = deconvolveDual(H, Ht, y, params)");
    }

    const mxArray* mex_H = prhs[0];
    const mxArray* mex_Ht = prhs[1];
    const mxArray* mex_y = prhs[2];
    const mxArray* mex_params = prhs[3];
    const int numDims = mxGetNumberOfDimensions(mex_y);
    if (numDims != 3)
        mexErrMsgTxt("Input error: y must be 3-dimensional array");
    const std::vector<int> matlabDims(mxGetDimensions(mex_y), mxGetDimensions(mex_y)+numDims);
    const std::vector<int> dims(matlabDims.rbegin(), matlabDims.rend());
    const int size = std::accumulate(begin(dims), end(dims), 1, [](double a, double b) { return a*b;});

    if (!mxIsStruct(mex_params))
        mexErrMsgTxt("Input error: params must be a struct");
    if (mxGetNumberOfElements(mex_params) != 1)
        mexErrMsgTxt("Input error: params must be 1x1 struct");
    
    DeconvolveParams params{};
    
    mxArray *mex_progress = mxGetField(mex_params, 0, "progress");
    double regularizerWeight = 10.0;
    double regularizerMax = 10.0;
    if (!mex_progress)
        mexErrMsgTxt("Input error: params.progress missing or bad value");
    {
        mxArray *mexField = mxGetField(mex_params, 0, "maxIter");
        if (mexField)
            params.maxIterations = mxGetPr(mexField)[0];
        else
            params.maxIterations = 0;

        mexField = mxGetField(mex_params, 0, "dataSmoothing");
        if (mexField)
            params.dataSmoothing = mxGetPr(mexField)[0];

        mexField = mxGetField(mex_params, 0, "smoothing");
        if (mexField)
            params.maxSmoothing = mxGetPr(mexField)[0];

        mexField = mxGetField(mex_params, 0, "minSmoothing");
        if (mexField)
            params.minSmoothing = mxGetPr(mexField)[0];

        mexField = mxGetField(mex_params, 0, "smoothWeight");
        if (mexField)
            regularizerWeight = mxGetPr(mexField)[0];

        mexField = mxGetField(mex_params, 0, "smoothMax");
        if (mexField)
            regularizerMax = mxGetPr(mexField)[0];
    }

    Array<3> y{dims};
    y.assign(mxGetPr(mex_y), mxGetPr(mex_y)+size);

    mxArray* callbackArray = mxCreateNumericArray(3, matlabDims.data(), mxDOUBLE_CLASS, mxREAL);

    int hEvaluations = 0;
    LinearSystem<3> H = [&](const Array<3>& x) -> Array<3> {
        mxArray* callbackRhs[] = { const_cast<mxArray*>(mex_H), callbackArray };
        mxArray* callbackLhs[] = { nullptr };
        double* argData = mxGetPr(callbackRhs[1]);
        for (int i = 0; i < size; ++i) 
            argData[i] = x.data()[i];
        mexCallMATLAB(1, callbackLhs, 2, callbackRhs, "feval");
        Array<3> result{dims};
        double *lhsData = mxGetPr(callbackLhs[0]);
        for (int i = 0; i < size; ++i)
            result.data()[i] = lhsData[i];
        mxDestroyArray(callbackLhs[0]);
        hEvaluations++;
        return result;
    };
    LinearSystem<3> Ht = [&](const Array<3>& x) -> Array<3> {
        mxArray* callbackRhs[] = { const_cast<mxArray*>(mex_Ht), callbackArray };
        mxArray* callbackLhs[] = { nullptr };
        double* argData = mxGetPr(callbackRhs[1]);
        for (int i = 0; i < size; ++i) 
            argData[i] = x.data()[i];
        mexCallMATLAB(1, callbackLhs, 2, callbackRhs, "feval");
        Array<3> result{dims};
        double *lhsData = mxGetPr(callbackLhs[0]);
        for (int i = 0; i < size; ++i)
            result.data()[i] = lhsData[i];
        mxDestroyArray(callbackLhs[0]);
        hEvaluations++;
        return result;
    };
    GridRegularizer<3> R{dims, 13, 4.0, regularizerMax, regularizerWeight};
    ProgressCallback<3> pc = [&](const Array<3>& x, double dual, double primalData, double primalReg, double smoothing) { 
        mxArray* callbackRhs[] = { 
            const_cast<mxArray*>(mex_progress), 
            callbackArray,
            mxCreateDoubleScalar(dual),
            mxCreateDoubleScalar(primalData),
            mxCreateDoubleScalar(primalReg),
            mxCreateDoubleScalar(smoothing)
        };
        double* argData = mxGetPr(callbackRhs[1]);
        for (int i = 0; i < size; ++i) 
            argData[i] = x.data()[i];
        mexCallMATLAB(0, nullptr, 6, callbackRhs, "feval");
        //throw DeconvolutionMexException{"Terminating in progress callback"};
    };
    DeconvolveStats stats{};

    try {
        auto x = Deconvolve<3>(y, H, Ht, R, pc, params, stats);
        plhs[0] = mxCreateNumericArray(3, matlabDims.data(), mxDOUBLE_CLASS, mxREAL);
        double* resultData = mxGetPr(plhs[0]);
        for (int i = 0; i < size; ++i) 
            resultData[i] = x.data()[i];
    } catch (const DeconvolutionMexException& e) {
        mexErrMsgIdAndTxt("Deconvolve:main", "Exception occurred: %s", e.what());
    } 

    mexPrintf("Iteration time:   %f\n", stats.iterTime);
    mexPrintf("Regularizer time: %f\n", stats.regularizerTime);
    mexPrintf("Data time:        %f\n", stats.dataTime);
    mexPrintf("Unary time:       %f\n", stats.unaryTime);
    mexPrintf("Evaluations of H: %d\n", hEvaluations);
}
