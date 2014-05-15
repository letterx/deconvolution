#include "quadratic-min.hpp"
#include "common.hpp"
#include "util.hpp"
#include <lbfgs.h>
#include <iostream>
#include "cg.h"

namespace deconvolution {

template <int D>
struct QuadraticMinData {
    const LinearSystem<D>& Q;
    const Array<D>& b;
    const Array<D>& x;
};

template <int D>
static double quadraticEvaluate(
        void *instance,
        const double *xData,
        double *g,
        const int n,
        const double step) {
    const auto* data = static_cast<QuadraticMinData<D>*>(instance);
    const auto& Q = data->Q;
    const auto& b = data->b;
    const auto& x = data->x;
    ASSERT(x.data() == xData);
    ASSERT(int(x.num_elements()) == n);

    auto Qx = Q(x);
    auto xQx = dot(x, Qx);
    auto bx = dot(b, x);
    for (int i = 0; i < n; ++i)
        g[i] = 2*Qx.data()[i] - b.data()[i];

    //std::cout << "Evaluate: " << xQx - bx << "\n";
    return xQx - bx;
}

int quadraticProgress(
        void *instance,
        const double *x,
        const double *g,
        const double fx,
        const double xnorm,
        const double gnorm,
        const double step,
        int n,
        int k,
        int ls) {
    /*
    printf("\tIteration %d:\n", k);
    printf("\t  fx = %f", fx);
    printf("\t  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    */
    return 0;
}

template <int D>
double quadraticMin(const LinearSystem<D>& Q, const Array<D>& b, Array<D>& x) {
    auto n = x.num_elements();
    lbfgs_parameter_t params;
    double fVal = 0;

    lbfgs_parameter_init(&params);
    QuadraticMinData<D> algData = {Q, b, x};

    auto retCode = lbfgs(n, x.origin(), &fVal, quadraticEvaluate<D>, quadraticProgress, &algData, &params);
    //std::cout << "\tFinished quadraticMin: " << retCode << ",\t" << fVal << "\n";
    if (retCode < 0)
        std::cerr << "*ERROR*: Finished quadraticMin with status " << retCode << "\n";
    return fVal;
}

template <int D>
class Vector {
    public:
        Vector() = default;
        explicit Vector(const Array<D>& data) : _data(data) { };
        Vector& operator=(const Vector& v) {
            if (_data.num_elements() != v._data.num_elements())
                _data.resize(std::vector<typename Array<D>::size_type>(v._data.shape(), v._data.shape()+D));
            _data = v._data;
            return *this;
        }

        Vector& operator=(double scalar) {
            for (auto& x : _data)
                x = scalar;
            return *this;
        };
        Vector& operator+=(const Vector& v) {
            _data += v.data();
            return *this;
        }
        Vector<D> operator+(const Vector& v) const {
            return Vector<D>{_data+v._data};
        }
        Vector& operator-=(const Vector& v) {
            _data -= v.data();
            return *this;
        }
        Vector<D> operator-(const Vector& v) const {
            return Vector<D>{_data-v._data};
        }
        double operator()(int idx) const {
            return _data.data()[idx];
        }
        Array<D>& data() { return _data; }
        const Array<D>& data() const { return _data; }
    protected:
        Array<D> _data;
};

template <int D>
Vector<D> operator*(double scalar, const Vector<D>& v) {
    return Vector<D>{scalar*v.data()};
}

template <int D>
double dot(const Vector<D>& v1, const Vector<D>& v2) {
    return dot(v1.data(), v2.data());
}

template <int D>
double norm(const Vector<D>& v) {
    return norm(v.data());
}

template <int D>
class Matrix {
    public:
        Matrix(const LinearSystem<D>& M, const LinearSystem<D>& Mt) : _M(M), _Mt(Mt) { };
        Vector<D> operator*(const Vector<D>& v) const {
            return Vector<D>{_M(v.data())};
        };
        Vector<D> trans_mult(const Vector<D>& v) const {
            return Vector<D>{_Mt(v.data())};
        };

    protected:
        const LinearSystem<D>& _M;
        const LinearSystem<D>& _Mt;
};

template <int D>
class Preconditioner {
    public:
        Vector<D> solve(const Vector<D>& v) const { return v; }
        Vector<D> trans_solve(const Vector<D>& v) const { return v; }
};

        

template <int D>
double quadraticMinCG(const LinearSystem<D>& Q, const Array<D>& b, Array<D>& x) {
    double tol = 1e-8;
    int maxIter = 100;
    auto M = Matrix<D>{Q, Q};
    auto B = Vector<D>{0.5*b};
    auto X = Vector<D>{x};
    auto P = Preconditioner<D>{};

    auto retCode = CG(M, X, B, P, maxIter, tol);
    if (retCode)
        std::cout << "*** CG reached maxIter --- residual: " << tol << "***\n";
    x = X.data();
    return dot(x, Q(x)) - dot(b, x);
}

#define INSTANTIATE_DECONVOLVE(d) \
    template double quadraticMinCG<d>(const LinearSystem<d>& Q, const Array<d>& b, Array<d>& x);

INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
