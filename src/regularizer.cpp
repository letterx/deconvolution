#include "regularizer.hpp"
#include <iostream>
#include <cfloat>
#include <mutex>
#include "tbb/tbb.h"
#include "util.hpp"
#include "transport.hpp"

namespace deconvolution {

static double logEpsilon = log(DBL_EPSILON);

void incrementBase(const std::vector<int>& extents, int subproblem, std::vector<int>& base) {
    for (int pos = 0; pos < int(extents.size()); ++pos) {
        if (pos == subproblem)
            continue;
        base[pos]++;
        if (base[pos] < extents[pos])
            return;
        else
            base[pos] = 0;
    }
}

template <int D, class EP>
double GridRegularizer<D, EP>::minMarginal(int subproblem, 
        const Array<D+1>& unaries,
        Array<D+1>& marginals) const {
    assert(subproblem >= 0 && subproblem < D);
    double objective = 0;
    std::mutex objectiveMutex;

    for (int i = 0; i < D+1; ++i) {
        assert(unaries.shape()[i] == marginals.shape()[i]);
        assert(unaries.index_bases()[i] == marginals.index_bases()[i]);
    }

    assert(unaries.strides()[D] == 1);
    assert(unaries.strides()[subproblem] == maxLabels());

    const int width = unaries.shape()[subproblem]; // Length along this dimension of the grid
    const int rowStride = width*_numLabels;
    const int numRows = unaries.num_elements() / rowStride;

    std::vector<double> m_L(_numLabels*width, 0);
    std::vector<double> m_R(_numLabels*width, 0);

    for (int i = 0; i < numRows; ++i) {
        const int baseVar = i*width;
        const int baseIdx = baseVar*_numLabels;
        const auto* unaries_i = unaries.data() + baseIdx;
        auto* marginals_i = marginals.data() + baseIdx;

        // Compute log m_L
        // Base step
        for (int l = 0; l < _numLabels; ++l) {
            m_L[l] = unaries_i[l];
        }
        // Inductive step
        for (int j = 1; j < width; ++j) {
            const int currVar = baseVar + j;
            for (int lCurr = 0; lCurr < _numLabels; ++lCurr) {
                double minMessage = std::numeric_limits<double>::max();
                for (int lPrev = 0; lPrev < _numLabels; ++lPrev) {
                    auto cost = _edgePotential.edgeFn(getLabel(currVar-1, lPrev), getLabel(currVar, lCurr))
                        + m_L[(j-1)*_numLabels+lPrev];
                    minMessage = std::min(minMessage, cost);
                }
                m_L[j*_numLabels+lCurr] = unaries_i[j*_numLabels+lCurr] + minMessage;
            }
        }

        // Compute log m_R
        for (int lCurr = 0; lCurr < _numLabels; ++lCurr) {
            m_R[(width-1)*_numLabels+lCurr] = 0.0;
        }
        for (int j = width-2; j >= 0; --j) {
            const int currVar = baseVar + j;
            for (int lCurr = 0; lCurr < _numLabels; ++lCurr) {
                double minMessage = std::numeric_limits<double>::min();
                for (int lPrev = 0; lPrev < _numLabels; ++lPrev) {
                    auto cost = _edgePotential.edgeFn(getLabel(currVar, lCurr), getLabel(currVar+1, lPrev))
                        + unaries_i[(j+1)*_numLabels + lPrev] + m_R[(j+1)*_numLabels+lPrev];
                        minMessage = std::min(minMessage, cost);
                }
                m_R[j*_numLabels+lCurr] = minMessage;
            }
        }

        // Compute marginals
        for (int j = 0; j < width; ++j) {
            for (int l = 0; l < _numLabels; ++l) {
                marginals_i[j*_numLabels+l] = m_L[j*_numLabels+l] + m_R[j*_numLabels+l];
            }
        }

        // Objective is min-marginal of any variable (we pick first one)
        double minMarginal = std::numeric_limits<double>::max();
        for (int l = 0; l < _numLabels; ++l)
            minMarginal = std::min(minMarginal, marginals_i[l]);

        {
            std::unique_lock<std::mutex> l(objectiveMutex);
            objective += minMarginal;
        }
    }
    return objective;
}

/*
 *template <int D, class EP>
 *double GridRegularizer<D, EP>::evaluate(int subproblem, const double* lambda_a, double smoothing, double* gradient, double* diagHessian) const {
 *    assert(subproblem >= 0 && subproblem < D);
 *    double objective = 0;
 *    std::mutex objectiveMutex;
 *
 *    std::vector<int> augmentedExtents(_extents);
 *    augmentedExtents.push_back(1);
 *    const auto extents = arrFromVec<D+1>(augmentedExtents);
 *    const auto strides = stridesFromExtents(extents);
 *    const int width = extents[subproblem]; // Length along this dimension of the grid
 *    const double smoothingMult = 1.0/smoothing;
 *
 *    std::array<int, D+1> order;
 *    for (int i = 0; i < D+1; ++i) order[i] = i;
 *    order = bringToFront(order, subproblem);
 *
 *    tbb::parallel_for(size_t(0), size_t(extents[order[1]]), [&](size_t i) {
 *    //for (size_t i = 0; i < size_t(extents[order[1]]); ++i) {
 *        int baseIdx = i*strides[order[1]];
 *        std::vector<double> lambdaSlice(_numLabels*width, 0);
 *        std::vector<double> m_L(_numLabels*width, 0);
 *        std::vector<double> m_R(_numLabels*width, 0);
 *        std::vector<double> logMarg(_numLabels, 0);
 *        std::vector<double> labelCosts(_numLabels, 0);
 *        std::vector<double> currLabels(_numLabels, 0);
 *        std::vector<double> prevLabels(_numLabels, 0);
 *
 *        double objective_i = 0;
 *
 *        arrayForEachTail<D+1, 2>(order, extents, strides, baseIdx, 
 *                [&](int baseIdx) -> void {
 *            int stride = strides[subproblem];
 *
 *            for (int j = 0; j < width; ++j)
 *                for (int l = 0; l < _numLabels; ++l)
 *                    lambdaSlice[j*_numLabels + l] = lambda_a[(baseIdx + j*stride)*_numLabels + l];
 *
 *            // Compute log m_L
 *            // Base step
 *            for (int l = 0; l < _numLabels; ++l) {
 *                m_L[l] = smoothingMult*lambdaSlice[l];
 *                currLabels[l] = _getLabel(baseIdx, l);
 *            }
 *            // Inductive step
 *            for (int j = 1; j < width; ++j) {
 *                int idx = baseIdx+j*stride;
 *                std::swap(currLabels, prevLabels);
 *                for (int lCurr = 0; lCurr < _numLabels; ++lCurr) {
 *                    currLabels[lCurr] = _getLabel(idx, lCurr);
 *                    double maxMessage = std::numeric_limits<double>::lowest();
 *                    for (int lPrev = 0; lPrev < _numLabels; ++lPrev) {
 *                        labelCosts[lPrev] = -_edgePotential.edgeFn(prevLabels[lPrev], currLabels[lCurr])*smoothingMult + m_L[(j-1)*_numLabels+lPrev];
 *                        maxMessage = std::max(maxMessage, labelCosts[lPrev]);
 *                    }
 *                    double sumExp = 0;
 *                    for (int lPrev = 0; lPrev < _numLabels; ++lPrev) {
 *                        double shiftedCost = labelCosts[lPrev] - maxMessage;
 *                        if (shiftedCost >= logEpsilon) // Don't take exp if result will be less than 1e-18
 *                            sumExp += exp(shiftedCost);
 *                    }
 *                    m_L[j*_numLabels+lCurr] = lambdaSlice[j*_numLabels+lCurr]*smoothingMult + maxMessage + log(sumExp);
 *                }
 *            }
 *
 *            // Compute log m_R
 *            for (int lCurr = 0; lCurr < _numLabels; ++lCurr) {
 *                m_R[(width-1)*_numLabels+lCurr] = 0.0;
 *                currLabels[lCurr] = _getLabel(baseIdx+(width-1)*stride, lCurr);
 *            }
 *            for (int j = width-2; j >= 0; --j) {
 *                int idx = baseIdx+j*stride;
 *                std::swap(currLabels, prevLabels);
 *                for (int lCurr = 0; lCurr < _numLabels; ++lCurr) {
 *                    currLabels[lCurr] = _getLabel(idx, lCurr);
 *                    double maxMessage = std::numeric_limits<double>::lowest();
 *                    for (int lPrev = 0; lPrev < _numLabels; ++lPrev) {
 *                        labelCosts[lPrev] = -(_edgePotential.edgeFn(currLabels[lCurr], prevLabels[lPrev]) - lambdaSlice[(j+1)*_numLabels+lPrev])*smoothingMult + m_R[(j+1)*_numLabels+lPrev];
 *                        maxMessage = std::max(maxMessage, labelCosts[lPrev]);
 *                    }
 *                    double sumExp = 0;
 *                    for (int lPrev = 0; lPrev < _numLabels; ++lPrev) {
 *                        double shiftedCost = labelCosts[lPrev] - maxMessage;
 *                        if (shiftedCost >= logEpsilon) // Don't take exp if result will be less than 1e-18
 *                            sumExp += exp(shiftedCost);
 *                    }
 *                    m_R[j*_numLabels+lCurr] = maxMessage + log(sumExp);
 *                }
 *            }
 *
 *            // Compute marginals, put them in G
 *            double logSumExp = 0;
 *            for (int j = 0; j < width; ++j) {
 *                double maxMarg = std::numeric_limits<double>::lowest();
 *                for (int l = 0; l < _numLabels; ++l) {
 *                    logMarg[l] = m_L[j*_numLabels+l] + m_R[j*_numLabels+l];
 *                    maxMarg = std::max(maxMarg, logMarg[l]);
 *                }
 *                double sumExp = 0;
 *                for (int l = 0; l < _numLabels; ++l)
 *                    sumExp += exp(logMarg[l] - maxMarg);
 *                logSumExp = maxMarg + log(sumExp);
 *                for (int l = 0; l < _numLabels; ++l) {
 *                    double grad_jl = -exp(logMarg[l] - logSumExp);
 *                    assert(0 <= -grad_jl && -grad_jl <= 1.0);
 *                    if (gradient)
 *                        gradient[(baseIdx + j*stride)*_numLabels + l] += grad_jl;
 *                    if (diagHessian)
 *                        diagHessian[(baseIdx + j*stride)*_numLabels + l] = smoothingMult * (-grad_jl)*(1 + grad_jl);
 *                }
 *            }
 *            objective_i += -smoothing*logSumExp;
 *        });
 *        {
 *            std::unique_lock<std::mutex> l(objectiveMutex);
 *            objective += objective_i;
 *        }
 *    });
 *    return objective;
 *}
 */

template <int D, class EP>
double GridRegularizer<D, EP>::primal(const double* x, double* gradient) const {
    double objective = 0;
    const auto X = boost::const_multi_array_ref<double, D>{x, _extents};

    for (int subproblem = 0; subproblem < D; ++subproblem) {
        int width = _extents[subproblem]; // Length along this dimension of the grid
        std::vector<int> base(D, 0);
        int numBases = 1;
        for (int i = 0; i < D; ++i)
            numBases *= (i == subproblem) ? 1 : _extents[i];

        for (int countBase = 0; countBase < numBases; ++countBase, incrementBase(_extents, subproblem, base)) {
            int baseIdx = 0;
            for (int i = 0; i < D; ++i) baseIdx += base[i]*X.strides()[i];
            int stride = X.strides()[subproblem];
            for (int i = 0; i < width-1; ++i) {
                auto idx1 = baseIdx+i*stride;
                auto idx2 = baseIdx+(i+1)*stride;
                objective += _edgePotential.edgeFn(x[idx1], x[idx2]);
                if (gradient)
                    _edgePotential.edgeGrad(x[idx1], x[idx2], gradient[idx1], gradient[idx2]);
            }
        }
    }
    return objective;
}

template <int D, class EP>
double GridRegularizer<D, EP>::fractionalPrimal(const std::vector<double>& primalMu_i) const {
    double objective = 0;
    // use X array just to compute strides consistently with other functions
    const auto X = boost::const_multi_array_ref<double, D>{nullptr, _extents};

    std::vector<double> costs(_numLabels*_numLabels);
    std::vector<double> flow(_numLabels*_numLabels);

    for (int subproblem = 0; subproblem < D; ++subproblem) {
        int width = _extents[subproblem]; // Length along this dimension of the grid
        std::vector<int> base(D, 0);
        int numBases = 1;
        for (int i = 0; i < D; ++i)
            numBases *= (i == subproblem) ? 1 : _extents[i];

        for (int countBase = 0; countBase < numBases; ++countBase, incrementBase(_extents, subproblem, base)) {
            int baseIdx = 0;
            for (int i = 0; i < D; ++i) baseIdx += base[i]*X.strides()[i];
            int stride = X.strides()[subproblem];
            for (int i = 0; i < width-1; ++i) {
                int var1 = baseIdx+i*stride;
                int var2 = var1 + stride;
                for (int l1 = 0; l1 < _numLabels; ++l1) {
                    for (int l2 = 0; l2 < _numLabels; ++l2) {
                        costs[l1*_numLabels+l2] 
                            = _edgePotential.edgeFn(_getLabel(var1, l1), _getLabel(var2, l2));
                    }
                }
                objective += solveTransport(_numLabels, _numLabels, costs.data(),
                        primalMu_i.data() + var1*_numLabels, primalMu_i.data() + var2*_numLabels, 
                        flow.data());
            }
        }
    }
    return objective;
}

template <int D, class EP>
void GridRegularizer<D, EP>::convexCombination(const std::vector<double>& primalMu_i, Array<D>& x) const {
    int n = x.num_elements();
    for (int i = 0; i < n; ++i) {
        double coord = 0;
        for (int l = 0; l < _numLabels; ++l)
            coord += primalMu_i[i*_numLabels+l]*_getLabel(i, l);
        x.data()[i] = coord;
    }
}

/*
 *template <int D, class EP>
 *void GridRegularizer<D, EP>::sampleLabels(const Array<D>& x, double scale) {
 *    for (int i = 0; i < D; ++i)
 *        assert(static_cast<int>(x.shape()[i]) == _extents[i]);
 *    int n = std::accumulate(_extents.begin(), _extents.end(), 1, [](int i, int j) { return i*j; });
 *    for (int i = 0; i < n; ++i) {
 *        double val = x.data()[i];
 *        for (int l = 0; l < _numLabels; ++l) {
 *            _labels[i*_numLabels+l] = val+scale*(l-(_numLabels-1)/2);
 *            //_labels[i*_numLabels+l] = std::min(_labels[i*_numLabels+l], _numLabels*_labelScale);
 *            //_labels[i*_numLabels+l] = std::max(_labels[i*_numLabels+l], 0.0);
 *        }
 *    }
 *    _labelScale = scale;
 *}
 */

#define INSTANTIATE_DECONVOLVE_REGULARIZER(d) \
    template class GridRegularizer<d, TruncatedL1>; \
    template class GridRegularizer<d, SmoothEdge>; \
    template class GridRegularizer<d, ConvexCombEdge<SmoothEdge, L2Edge>>;
INSTANTIATE_DECONVOLVE_REGULARIZER(1)
INSTANTIATE_DECONVOLVE_REGULARIZER(2)
INSTANTIATE_DECONVOLVE_REGULARIZER(3)
#undef INSTANTIATE_DECONVOLVE_REGULARIZER

}
