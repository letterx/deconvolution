#include "deconvolve.hpp"
#include "quadratic-min.hpp"
#include "util.hpp"
#include "regularizer.hpp"
#include "optimal-grad.hpp"
#include <limits>
#include <iostream>
#include <chrono>
#include <random>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#include "optimization.h"
#pragma clang diagnostic pop

namespace deconvolution{

using namespace alglib;

template <int D>
struct DeconvolveData {
    Array<D>& x;
    const Array<D>& b;
    const LinearSystem<D>& Q;
    const Regularizer<D>& R;
    int numLambda;
    double constantTerm;
    real_1d_array& diagHessian;
    ProgressCallback<D>& pc;
    DeconvolveParams& params;
    DeconvolveStats& stats;
    std::function<double(const Array<D>&)> primalFn;
    int& totalIters;
    minlbfgsstate& lbfgsState;
};

template <int D>
double evaluateUnary(const double* dualVars, const DeconvolveData<D>& data, double* grad, double* diagHessian) {
    const auto& R = data.R;
    const auto numPrimalVars = int(data.x.num_elements());
    const auto numSubproblems = R.numSubproblems();
    const auto numLabels = R.numLabels();
    const auto numPerSubproblem = numPrimalVars*numLabels;
    const auto numLambda = data.numLambda;
    const double* nu = dualVars + numLambda;
    const double t = data.params.smoothing;

    double unaryObjective = 0;
    auto minTable = std::vector<double>(numLabels, 0);
    for (int i = 0; i < numPrimalVars; ++i) {
        auto nu_i = nu[i];
        double minValue = std::numeric_limits<double>::max();
        for (int xi = 0; xi < numLabels; ++xi) {
            double lambdaSum = 0;
            for (int alpha = 0; alpha < numSubproblems; ++alpha) 
                lambdaSum += dualVars[alpha*numPerSubproblem+i*numLabels+xi];
            minTable[xi] = lambdaSum + nu_i*R.getLabel(i, xi);
            minValue = std::min(minValue, minTable[xi]);
        }
        double expSum = 0;
        double nuGrad = 0;
        double nuDiagH = 0;
        for (int xi = 0; xi < numLabels; ++xi) {
            minTable[xi] = exp(-(minTable[xi]-minValue)/t);
            expSum += minTable[xi];
            nuGrad += minTable[xi]*R.getLabel(i, xi);
            nuDiagH += minTable[xi]*R.getLabel(i,xi)*R.getLabel(i,xi);
        }
        unaryObjective += minValue - t*log(expSum);
        nuGrad /= expSum;
        nuDiagH /= expSum;
        grad[i+numLambda] += nuGrad;
        assert(nuDiagH - nuGrad*nuGrad >= 0);
        diagHessian[i+numLambda] += 1.0/t * (nuDiagH - nuGrad*nuGrad);
        for (int alpha = 0; alpha < numSubproblems; ++alpha) {
            for (int xi = 0; xi < numLabels; ++xi) {
                const double gradLambda = minTable[xi]/expSum;
                assert(0 <= gradLambda && gradLambda <= 1.0);
                grad[alpha*numPerSubproblem+i*numLabels+xi] += gradLambda;
                diagHessian[alpha*numPerSubproblem+i*numLabels+xi] += 1.0/t * gradLambda * (1.0 - gradLambda);
            }
        }
    }
    return unaryObjective;
}

template <int D>
static void lbfgsEvaluate(
        const real_1d_array& lbfgsX,
        double& objective,
        real_1d_array& lbfgsGrad,
        void* instance) {
    const double* dualVars = lbfgsX.getcontent();
    double* grad = lbfgsGrad.getcontent();
    const int n = lbfgsX.length();
    auto* data = static_cast<DeconvolveData<D>*>(instance);
    auto& x = data->x;
    const auto& b = data->b;
    const auto& Q = data->Q;
    const auto& R = data->R;
    const double constantTerm = data->constantTerm;
    const auto numPrimalVars = int(x.num_elements());
    const auto numLabels = R.numLabels();
    const auto numPerSubproblem = numPrimalVars*numLabels;
    const auto numLambda = data->numLambda;
    const double* nu = dualVars + numLambda;
    const double t = data->params.smoothing;
    const double dataSmoothing = data->params.dataSmoothing;
    real_1d_array& diagHessian = data->diagHessian;
    DeconvolveStats& stats = data->stats;

    typedef std::chrono::duration<double> Duration;
    typedef std::chrono::system_clock Clock;

    auto iterStartTime = Clock::now();

    auto bPlusNu = b;
    for (int i = 0; i < numPrimalVars; ++i)
        bPlusNu.data()[i] += nu[i];

    for (int i = 0; i < n; ++i)
        grad[i] = 0;

    auto startTime = Clock::now();
    double regularizerObjective = 0;
    for (int i = 0; i < R.numSubproblems(); ++i)
        regularizerObjective += R.evaluate(i, dualVars+i*numPerSubproblem, t, grad+i*numPerSubproblem, diagHessian.getcontent()+i*numPerSubproblem);
    stats.regularizerTime += Duration{Clock::now() - startTime}.count();

    startTime = Clock::now();
    double dataObjective = quadraticMinCG<D>(Q, bPlusNu, x) + constantTerm;
    for (int i = 0; i < numPrimalVars; ++i) {
        grad[i+numLambda] = -x.data()[i];
        diagHessian[i+numLambda] = 1/(2*dataSmoothing);
    }
    stats.dataTime += Duration{Clock::now() - startTime}.count();

    startTime = Clock::now();
    double unaryObjective = evaluateUnary(dualVars, *data, grad, diagHessian.getcontent());
    stats.unaryTime += Duration{Clock::now() - startTime}.count();

    for (int i = 0; i < n; ++i)
        grad[i] = -grad[i];
    objective = -(regularizerObjective + dataObjective + unaryObjective);
    //std::cout << "Evaluate: " << objective << "\t(" << regularizerObjective << ", " << dataObjective << ", " << unaryObjective << ")\n";
    
    for (int i = 0; i < n; ++i) {
        diagHessian[i] = std::max(diagHessian[i], 1e-7);
    }
    minlbfgssetprecdiag(data->lbfgsState, diagHessian);

    stats.iterTime += Duration{Clock::now() - iterStartTime}.count();

}

template <int D>
static void lbfgsProgress(
        const real_1d_array& lbfgsX,
        double fx,
        void *instance) {
    auto* data = static_cast<DeconvolveData<D>*>(instance);
    
    data->totalIters++;

    double primalData = data->primalFn(data->x);
    double primalReg  = data->R.primal(data->x.data());
    double primal = primalData + primalReg;

    std::cout << "Deconvolve Iteration " << data->totalIters << "\t";
    std::cout << "dual: " << -fx << "\tprimal: " << primal << "\n";
    data->pc(data->x, -fx, primalData, primalReg, data->params.smoothing);
}

/*
template <int D>
double estimateQDiag(const LinearSystem<D>& Q, const Array<D>& x) {
    std::cout << "Estimating Q Diagonal\n";
    const int repetitions = 10;
    std::mt19937 generator;
    std::uniform_int_distribution<int> dist(0, x.num_elements()-1);
    auto rand = std::bind(dist, generator);
    double estimate = 0;
    double sumSquares = 0;
    for (int i = 0; i < repetitions; ++i) {
        Array<D> ei = x;
        for (int idx = 0; idx < static_cast<int>(ei.num_elements()); ++idx)
            ei.data()[idx] = 0.0;
        const int randIdx = rand();
        ei.data()[randIdx] = 1.0;
        double diag = dot(ei, Q(ei));
        estimate += diag;
        sumSquares += diag*diag;
    }
    estimate /= repetitions;
    sumSquares /= repetitions;
    std::cout << "\tAverage: " << estimate << "\tStdDev: " << sqrt(sumSquares - estimate*estimate) << "\n";
    return estimate;
}
*/

template <int D>
Array<D> Deconvolve(const Array<D>& y, 
        const LinearSystem<D>& H, 
        const LinearSystem<D>& Ht, 
        Regularizer<D>& R, 
        ProgressCallback<D>& pc, 
        DeconvolveParams& params,
        DeconvolveStats& stats) {
    constexpr double datascale = 1;
    Array<D> b = 2*datascale*Ht(y);
    Array<D> x = b;
    LinearSystem<D> Q = [&](const Array<D>& x) -> Array<D> { return datascale*Ht(H(x)) + params.dataSmoothing*x; };
    double constantTerm = datascale*dot(y, y);
    std::function<double(const Array<D>& x)> primalFn = 
        [&](const Array<D>& x) -> double {
            auto res = H(x) - y;
            return dot(res, res) + params.dataSmoothing*dot(x, x);
        };


    int numPrimalVars = x.num_elements();
    int numLambda = numPrimalVars*R.numSubproblems()*R.numLabels();
    int numDualVars = numLambda + numPrimalVars; // Including all lambda vars + vector nu
    auto dualVars = std::unique_ptr<double>(new double[numDualVars]);
    double* lambda = dualVars.get();
    double* nu = dualVars.get() + numLambda;
    for (size_t i = 0; i < x.num_elements(); ++i) {
        x.data()[i] = 0;
    }
    std::cout << "Finding least-squares fit\n";
    quadraticMinCG<D>(Q, b, x);
    R.sampleLabels(x, 0.5);

    for (int i = 0; i < numPrimalVars; ++i)
        //nu[i] = -0.00001*x.data()[i];
        nu[i] = 0;
    for (int sp = 0; sp < R.numSubproblems(); ++sp) {
        for (int i = 0; i < numPrimalVars; ++i) {
            for (int l = 0; l < R.numLabels(); ++l) {
                lambda[sp*numPrimalVars*R.numLabels() + i*R.numLabels() + l] = -nu[i]*R.getLabel(i, l)/R.numSubproblems();
            }
        }
    }

    double fVal = 0;

    real_1d_array lbfgsX;
    lbfgsX.setcontent(numDualVars, dualVars.get());

    real_1d_array diagHessian;
    diagHessian.setlength(numDualVars);

    minlbfgsstate lbfgsState;
    minlbfgsreport lbfgsReport;

    minlbfgscreate(10, lbfgsX, lbfgsState);
    minlbfgssetxrep(lbfgsState, true);

    std::cout << "Begin lbfgs\n";
    int totalIters = 0;
    for (int samplingIter = 0; samplingIter < 1; ++samplingIter) {
        if (totalIters >= params.maxIterations)
            break;
        for (; params.smoothing >= params.minSmoothing; params.smoothing /= 2) {
            if (totalIters >= params.maxIterations)
                break;
            std::cout << "\t*** Smoothing: " << params.smoothing << " ***\n";
            auto algData = DeconvolveData<D>{x, b, Q, R, numLambda, constantTerm, diagHessian, pc, params, stats, primalFn, totalIters, lbfgsState};

            minlbfgssetcond(lbfgsState, 1.0, 0.000001, 0, params.maxIterations - totalIters);
            minlbfgsrestartfrom(lbfgsState, lbfgsX);
            minlbfgsoptimize(lbfgsState, lbfgsEvaluate<D>, lbfgsProgress<D>, &algData);
            minlbfgsresults(lbfgsState, lbfgsX, lbfgsReport);

            double primal = primalFn(x) + R.primal(x.data());
            if (primal < -fVal) break;
            std::cout << "\tL-BFGS finished\n";
        }
        std::cout << "*** Resampling ***\n";
        R.sampleLabels(x, 1.0/((samplingIter+1)*(samplingIter+1)));
    }

    
    //auto retCode = optimalGradDescent(numDualVars, dualVars.get(), &fVal, deconvolveEvaluate<D>, deconvolveProgress<D>, &algData);

    return x;
}

template <int D>
void ADMMSubproblemData(DeconvolveParams& params, 
        const LinearSystem<D>& Q, const Array<D>& b,
        const Regularizer<D>& R,
        const std::vector<double>& nu, 
        std::vector<double>& mu1, const std::vector<double>& mu2) {
    // Matrix T and vector c are the quadratic and linear terms for the 
    // quadratic program in the ADMM subproblem
    int n = b.num_elements();

    std::vector<double> normLi(n);
    for (int i = 0; i < n; ++i) {
        for (int l = 0; l < R.numLabels(); ++l) {
            double label = R.getLabel(i, l);
            normLi[i] += label*label;
        }
    }
    const LinearSystem<D>& T = [&](const Array<D>& x) -> Array<D> {
        Array<D> result = Q(x);
        assert(static_cast<int>(x.num_elements()) == n);
        for (int i = 0; i < n; ++i)
            result.data()[i]  += params.admmRho * x.data()[i] / (normLi[i] * 2);
        return result;
    };

    std::vector<double> dotMuLi(n);
    for (int i = 0; i < n; ++i) {
        dotMuLi[i] = 0;
        for (int l = 0; l < R.numLabels(); ++l) {
            int muIdx = i*R.numLabels() + l;
            double label = R.getLabel(i, l);
            dotMuLi[i] += (mu2[muIdx] - nu[muIdx])*label;
        }
    }

    Array<D> c = b;
    for (int i = 0; i < n; ++i) {
        c.data()[i] += dotMuLi[i] / normLi[i];
    }

    // Initialize x to be current conve combination given by mu1
    Array<D> x = b;
    for (int i = 0; i < n; ++i) {
        x.data()[i] = 0;
        for (int l = 0; l < R.numLabels(); ++l) 
            x.data()[i] += mu1[i*R.numLabels() + l] * R.getLabel(i, l);
    }

    // Solve quadratic system
    quadraticMinCG<D>(T, c, x);

    for (int i = 0; i < n; ++i) {
        double liCoeff = (dotMuLi[i]/params.admmRho - x.data()[i]) / normLi[i];
        for (int l = 0; l < R.numLabels(); ++l) {
            int muIdx = i*R.numLabels() + l;
            mu1[muIdx] -= nu[muIdx]/params.admmRho + liCoeff * R.getLabel(i, l);
        }
    }
}

template <int D>
void ADMMSubproblemReg(DeconvolveParams& params, 
        Regularizer<D>& R,
        const std::vector<double>& nu, 
        const std::vector<double>& mu1, std::vector<double>& mu2) {

}

template <int D>
Array<D> DeconvolveADMM(const Array<D>& y, 
        const LinearSystem<D>& H, 
        const LinearSystem<D>& Ht, 
        Regularizer<D>& R, 
        ProgressCallback<D>& pc, 
        DeconvolveParams& params,
        DeconvolveStats& stats) {
    Array<D> b = 2*Ht(y);
    Array<D> x = b;
    LinearSystem<D> Q = [&](const Array<D>& x) -> Array<D> { return Ht(H(x)); };
    LinearSystem<D> Qreg = [&](const Array<D>& x) -> Array<D> { return Ht(H(x)) + 0.001*x; };
    //double constantTerm = dot(y, y);
    std::function<double(const Array<D>& x)> dataFn = 
        [&](const Array<D>& x) -> double {
            auto res = H(x) - y;
            return dot(res, res);
        };


    const int numXVars = x.num_elements();
    const int numMu = numXVars*R.numLabels();
    auto mu1 = std::vector<double>(numMu);
    auto mu2 = std::vector<double>(numMu);
    auto nu = std::vector<double>(numMu);

    // Initialize primal variables to uniform probability on each label
    const double recipNumLabels = 1.0/R.numLabels();
    for (int i = 0; i < numMu; ++i) {
        mu1[i] = mu2[i] = recipNumLabels;
        nu[i] = 0;
    }

    std::cout << "Finding least-squares fit\n";
    quadraticMinCG<D>(Qreg, b, x);
    R.sampleLabels(x, 0.5);

    double resNorm = std::numeric_limits<double>::max();
    while (resNorm > params.admmConvergenceNorm) {
        ADMMSubproblemData<D>(params, Q, b, R, nu, mu1, mu2);
        ADMMSubproblemReg<D>(params, R, nu, mu1, mu2);
        resNorm = 0.0;
        for (int i = 0; i < numMu; ++i) {
            auto res = mu1[i] - mu2[i];
            nu[i] += params.admmRho*res;
            resNorm += res*res;
        }
    }
    return x;
}


#define INSTANTIATE_DECONVOLVE(d) \
    template Array<d> Deconvolve<d>(const Array<d>& y, const LinearSystem<d>& H, const LinearSystem<d>& Q, Regularizer<d>& R, ProgressCallback<d>& pc, DeconvolveParams& params, DeconvolveStats& s); \
    template Array<d> DeconvolveADMM<d>(const Array<d>& y, const LinearSystem<d>& H, const LinearSystem<d>& Q, Regularizer<d>& R, ProgressCallback<d>& pc, DeconvolveParams& params, DeconvolveStats& s);

INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
