#ifndef _DECONVOLVE_REGULARIZER_HPP_
#define _DECONVOLVE_REGULARIZER_HPP_

#include "deconvolve.hpp"
#include <vector>
#include <functional>
#include <assert.h>

namespace deconvolution {
template <int D> 
class Regularizer {
    public:
        Regularizer() { };

        virtual int numSubproblems() const = 0;
        virtual int numLabels() const = 0;
        virtual double getLabel(int var, int l) const = 0;
        virtual double evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient) const = 0;
};

template <int D>
class DummyRegularizer : public Regularizer<D> {
    public:
        DummyRegularizer() { };
        virtual int numSubproblems() const override { return 0; }
        virtual int numLabels() const override { return 2; }
        virtual double getLabel(int var, int l) const override { return l == 0 ? 0 : 255; }
        virtual double evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient) const override 
            { return 0; }
    private:
};

template <int D>
class GridRegularizer : public Regularizer<D> {
    public:
        typedef std::function<double(int, int)> EdgeFn;
        GridRegularizer(const std::vector<int>& extents, int numLabels, double labelScale, const EdgeFn& edgeFn) 
            : _extents(extents)
            , _numLabels(numLabels)
            , _labelScale(labelScale)
            , _edgeFn(edgeFn)
        { 
            assert(_extents.size() == D);
        }

        virtual int numSubproblems() const override { return D; }
        virtual int numLabels() const override { return _numLabels; }
        virtual double getLabel(int var, int l) const override { return l*_labelScale; }
        virtual double evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient) const override;
    private:
        std::vector<int> _extents;
        int _numLabels;
        double _labelScale;
        EdgeFn _edgeFn;
};
}


#endif
