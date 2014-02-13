#ifndef _DECONVOLVE_REGULARIZER_HPP_
#define _DECONVOLVE_REGULARIZER_HPP_

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
}


#endif
