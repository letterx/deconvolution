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
        virtual double evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient, double* diagHessian) const = 0;
        virtual double primal(const double* x, double* gradient) const = 0;
        virtual double fractionalPrimal(const std::vector<double>& primalMu_i) const = 0;
        virtual void sampleLabels(const Array<D>& x, double scale) { };
        virtual void convexCombination(const std::vector<double>& primalMu_i, Array<D>& x) const = 0;
};

template <int D>
class DummyRegularizer : public Regularizer<D> {
    public:
        DummyRegularizer() { };
        virtual int numSubproblems() const override { return 0; }
        virtual int numLabels() const override { return 2; }
        virtual double getLabel(int var, int l) const override { return l == 0 ? 0 : 255; }
        virtual double evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient, double* diagHessian) const override 
            { return 0; }
        virtual double primal(const double* x, double* gradient) const override { return 0; }
        virtual double fractionalPrimal(const std::vector<double>& primalMu_i) const override { return 0; }
        virtual void convexCombination(const std::vector<double>& primalMu_i, Array<D>& x) const override { }
    private:
};

template <int D>
class SmoothRegularizer : public Regularizer<D> {
    public:
        SmoothRegularizer(const std::vector<int>& extents, double smoothMax, double smoothWidth);

        virtual int numSubproblems() const override { return D; }
        virtual int numLabels() const override { return 1; }
        virtual double getLabel(int ver, int l) const override { return 0.0; }
};

class TruncatedL1 {
    public:
        TruncatedL1(double smoothMax, double smoothWeight)
            : _smoothMax(smoothMax)
            , _smoothWeight(smoothWeight)
        { }

        double edgeFn(double l1, double l2) const {
            return _smoothWeight*std::min(_smoothMax, fabs(l1 - l2));
        }
        void edgeGrad(double l1, double l2, double& g1, double& g2) const {
            auto diff = fabs(l1 - l2);
            if (diff > _smoothMax) {
                return;
            } else if (l1 > l2) {
                g1 += _smoothWeight;
                g2 += -_smoothWeight;
            } else {
                g1 += -_smoothWeight;
                g2 += _smoothWeight;
            }
        }

    private:
        double _smoothMax;
        double _smoothWeight;
};

class SmoothEdge {
    public:
        SmoothEdge(double weight, double width)
            : _weight(weight)
            , _recipWidth(1.0/width)
        { }

        double edgeFn(double l1, double l2) const {
            auto diff = l1 - l2;
            return _weight*(1.0 - 1.0/(1.0 + _recipWidth*diff*diff));
        }
        void edgeGrad(double l1, double l2, double& g1, double& g2) const {
            auto diff = l1 - l2;
            auto denom = 1.0 + _recipWidth*diff*diff;
            auto grad = _weight*2*_recipWidth*diff / (denom*denom);
            g1 += grad;
            g2 += -grad;
        }

    private:
        double _weight;
        double _recipWidth;
};

template <int D, class EdgePotential>
class GridRegularizer : public Regularizer<D> {
    public:
        GridRegularizer(const std::vector<int>& extents, int numLabels, double labelScale, const EdgePotential& edgePotential)
            : _extents(extents)
            , _numLabels(numLabels)
            , _labelScale(labelScale)
            , _labels(std::accumulate(extents.begin(), extents.end(), 1, [](int a, int b) { return a*b; })*numLabels, 0)
            , _edgePotential(edgePotential)
        { 
            assert(_extents.size() == D);
            int n = std::accumulate(extents.begin(), extents.end(), 1, [](int a, int b) { return a*b; });
            for (int i = 0; i < n; ++i)
                for (int l = 0; l < _numLabels; ++l)
                    _labels[i*_numLabels+l] = l*_labelScale;
        }
    private:
        // Internal non-virtual functions to improve inlining
        double _getLabel(int var, int l) const { return _labels[var*_numLabels+l]; }

    public:
        virtual int numSubproblems() const override { return D; }
        virtual int numLabels() const override { return _numLabels; }
        virtual double getLabel(int var, int l) const override { return _getLabel(var, l); }
        virtual double evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient, double* diagHessian) const override;
        virtual double primal(const double* x, double* gradient) const override;
        virtual double fractionalPrimal(const std::vector<double>& primalMu_i) const override;
        virtual void sampleLabels(const Array<D>& x, double scale) override;
        virtual void convexCombination(const std::vector<double>& primalMu_i, Array<D>& x) const override;
    protected:
        std::vector<int> _extents;
        int _numLabels;
        double _labelScale;
        std::vector<double> _labels;
        EdgePotential _edgePotential;
};

template <int D, class EdgePotential>
class GridRangeRegularizer : public GridRegularizer<D, EdgePotential> {
    public:
        GridRangeRegularizer(const std::vector<int>& extents,
                int numLabels,
                double labelScale,
                const EdgePotential& edgePotential,
                double maxLabel)
            : GridRegularizer<D, EdgePotential>(extents, numLabels, labelScale, edgePotential)
            , _maxLabel(maxLabel)
        { }

        virtual void sampleLabels(const Array<D>& x, double scale) override { }
        virtual double getLabel(int var, int l) const override {
            return _maxLabel*(static_cast<double>(l)/static_cast<double>(this->_numLabels));
        }

    protected:
        double _maxLabel;
};

}


#endif
