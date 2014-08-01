#ifndef _CONVEX_FN_HPP_
#define _CONVEX_FN_HPP_

#include <assert.h>
#include <vector>

class ConvexFn;

class PiecewiseLinearFn {
    public:
        template <typename Iter1, typename Iter2>
        PiecewiseLinearFn(Iter1&& xBegin, Iter1&& xEnd, Iter2&& fBegin)
            : _x(xBegin, xEnd)
            , _fx(fBegin, fBegin + (xEnd - xBegin))
        {
            const int n = _x.size();
            for (int i = 0; i < n-1; ++i) {
                assert(_x[i+1] > _x[i]);
                auto run = _x[i+1] - _x[i];
                auto rise = _fx[i+1] - _fx[i];
                _slope.push_back(rise / run);
            }
            _slope.push_back(std::numeric_limits<double>::max());
        }

        double operator()(double x) const;
        ConvexFn convexify() const;

    protected:
        std::vector<double> _x;
        std::vector<double> _fx;
        std::vector<double> _slope;
};

class ConvexFn : public PiecewiseLinearFn {
    public:
        template <typename Iter1, typename Iter2>
        ConvexFn(Iter1&& xBegin, Iter1&& xEnd, Iter2&& fBegin)
            : PiecewiseLinearFn(xBegin, xEnd, fBegin)
        { }

        double moreauEnvelope(double x, double t) const;

    protected:
        double moreauY(double x, double t) const;
};

inline double PiecewiseLinearFn::operator()(double x) const {
    const int n = _x.size();
    if (x < _x[0]) return std::numeric_limits<double>::max();
    for (int i = 1; i < n; ++i) {
        if (x <= _x[i]) {
            auto diff = x - _x[i-1];
            return _fx[i-1] + diff*_slope[i-1];
        }
    }
    assert(x > _x[n-1]);
    return std::numeric_limits<double>::max();
}

inline ConvexFn PiecewiseLinearFn::convexify() const { 
    const int n = _x.size();
    assert(n > 1);

    auto xs = std::vector<double>{
        std::numeric_limits<double>::lowest(),
        _x[0]};
    auto fx = std::vector<double>{
        std::numeric_limits<double>::lowest(),
        _fx[0]};
    auto slopes = std::vector<double>{std::numeric_limits<double>::lowest()};

    for (int i = 1; i < n; ++i) {
        double lastSlope = slopes.back();
        double nextX = _x[i];
        double nextF = _fx[i];
        while (true) {
            double rise = nextF - fx.back();
            double run = nextX - xs.back();
            double s = rise/run;
            if (s <= lastSlope) {
                xs.pop_back();
                fx.pop_back();
                slopes.pop_back();
                assert(xs.size() >= 2 && fx.size() >= 2 && slopes.size() >= 1);
            } else {
                xs.push_back(nextX);
                fx.push_back(nextF);
                slopes.push_back(s);
                break;
            }
        }
    }
    return ConvexFn{xs.begin(), xs.end(), fx.begin()};
}

inline double ConvexFn::moreauY(double x, double t) const {
    const int n = _x.size();
    for (int i = 0; i < n-1; ++i) {
        if (x <= _x[i] + t*_slope[i]) return _x[i];
        if (x <= _x[i+1] + t*_slope[i]) return x - t*_slope[i];
    }
    return _x[n-1];
}

inline double ConvexFn::moreauEnvelope(double x, double t) const {
    auto y = moreauY(x, t);
    auto dist = y - x;
    return (*this)(y) + 0.5*dist*dist/t;
}

#endif
