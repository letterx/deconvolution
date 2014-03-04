#include "deconvolve.hpp"
#include "quadratic-min.hpp"
#include "util.hpp"
#include "regularizer.hpp"
#include "optimal-grad.hpp"
#include <lbfgs.h>
#include <limits>
#include <iostream>
#include <chrono>

namespace deconvolution{

template <int D>
struct DeconvolveData {
    Array<D>& x;
    const Array<D>& b;
    const LinearSystem<D>& Q;
    const Regularizer<D>& R;
    int numLambda;
    double constantTerm;
    double lambdaScale;
    ProgressCallback<D>& pc;
    DeconvolveParams& params;
    DeconvolveStats& stats;
    std::function<double(const Array<D>&)> primalFn;
    int& totalIters;
};

template <int D>
static double deconvolveEvaluate(
        void* instance,
        const double* dualVars, 
        double* grad,
        const int n,
        const double step) {
    auto* data = static_cast<DeconvolveData<D>*>(instance);
    auto& x = data->x;
    const auto& b = data->b;
    const auto& Q = data->Q;
    const auto& R = data->R;
    const double constantTerm = data->constantTerm;
    const auto numPrimalVars = int(x.num_elements());
    const auto numSubproblems = R.numSubproblems();
    const auto numLabels = R.numLabels();
    const auto numPerSubproblem = numPrimalVars*numLabels;
    const auto numLambda = data->numLambda;
    const double* nu = dualVars + numLambda;
    const double t = data->params.smoothing;
    const double lambdaScale = data->lambdaScale;
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
        regularizerObjective += R.evaluate(i, dualVars+i*numPerSubproblem, t, lambdaScale, grad+i*numPerSubproblem);
    stats.regularizerTime += Duration{Clock::now() - startTime}.count();

    startTime = Clock::now();
    double dataObjective = quadraticMinCG<D>(Q, bPlusNu, x) + constantTerm;
    for (int i = 0; i < numPrimalVars; ++i)
        grad[i+numLambda] = -x.data()[i];
    stats.dataTime += Duration{Clock::now() - startTime}.count();

    startTime = Clock::now();
    double unaryObjective = 0;
    auto minTable = std::vector<double>(numLabels, 0);
    for (int i = 0; i < numPrimalVars; ++i) {
        auto nu_i = nu[i];
        double minValue = std::numeric_limits<double>::max();
        for (int xi = 0; xi < numLabels; ++xi) {
            double lambdaSum = 0;
            for (int alpha = 0; alpha < numSubproblems; ++alpha) 
                lambdaSum += dualVars[alpha*numPerSubproblem+i*numLabels+xi];
            minTable[xi] = lambdaScale*lambdaSum + nu_i*R.getLabel(i, xi);
            minValue = std::min(minValue, minTable[xi]);
        }
        double expSum = 0;
        double nuGrad = 0;
        for (int xi = 0; xi < numLabels; ++xi) {
            minTable[xi] = exp(-(minTable[xi]-minValue)/t);
            expSum += minTable[xi];
            nuGrad += minTable[xi]*R.getLabel(i, xi);
        }
        unaryObjective += minValue - t*log(expSum);
        grad[i+numLambda] += nuGrad/expSum;
        for (int alpha = 0; alpha < numSubproblems; ++alpha)
            for (int xi = 0; xi < numLabels; ++xi) 
                grad[alpha*numPerSubproblem+i*numLabels+xi] += lambdaScale*minTable[xi]/expSum;
    }
    stats.unaryTime += Duration{Clock::now() - startTime}.count();

    for (int i = 0; i < n; ++i)
        grad[i] = -grad[i];
    double objective = -(regularizerObjective + dataObjective + unaryObjective);
    std::cout << "Evaluate: " << objective << "\t(" << regularizerObjective << ", " << dataObjective << ", " << unaryObjective << ")\n";

    stats.iterTime += Duration{Clock::now() - iterStartTime}.count();

    return objective;
}

template <int D>
static int deconvolveProgress(
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
    auto* data = static_cast<DeconvolveData<D>*>(instance);
    //auto& x = data->x;
    const auto numPrimalVars = int(data->x.num_elements());
    const auto numLambda = data->numLambda;
    const double* nu = x + numLambda;
    const double* nuGrad = g + numLambda;
    
    data->totalIters++;

    double lambdaNorm = 0;
    double lambdaL1 = 0;
    double lambdaGradNorm = 0;
    for (int i = 0; i < numLambda; ++i) {
        lambdaNorm += x[i]*x[i];
        lambdaL1 += fabs(x[i]);
        lambdaGradNorm += g[i]*g[i];
    }
    lambdaNorm = sqrt(lambdaNorm);
    lambdaGradNorm = sqrt(lambdaGradNorm);

    double nuNorm = 0;
    double nuL1 = 0;
    double nuGradNorm = 0;
    for (int i = 0; i < numPrimalVars; ++i) {
        nuNorm += nu[i];
        nuL1 += fabs(nu[i]);
        nuGradNorm += nuGrad[i]*nuGrad[i];
    }
    nuNorm = sqrt(nuNorm);
    nuGradNorm = sqrt(nuGradNorm);

    double primalData = data->primalFn(data->x);
    double primalReg  = data->R.primal(data->x.data());
    double primal = primalData + primalReg;

    std::cout << "Deconvolve Iteration " << data->totalIters << "\n";
    std::cout << "\tf(x): " << -fx << "\tprimal: " << primal << "\txnorm: " << xnorm << "\tgnorm: " << gnorm << "\tstep: " << step << "\n";
    std::cout << "\t||lambda||: " << lambdaNorm << "\t||lambda||_1: " << lambdaL1 << "\t||Grad lambda||: " << lambdaGradNorm << "\n";
    std::cout << "\t||nu||:     " << nuNorm     << "\t||nu||_1:     " << nuL1     << "\t||Grad nu||:     " << nuGradNorm << "\n";
    std::cout << "\n";
    data->pc(data->x, -fx, primalData, primalReg, data->params.smoothing);
    return 0;
}

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
    R.sampleLabels(x, 1.0);

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

    lbfgs_parameter_t lbfgsParams;
    double fVal = 0;
    lbfgs_parameter_init(&lbfgsParams);
    //lbfgsParams.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
    //lbfgsParams.delta = 0.00001;
    //lbfgsParams.past = 100;
    lbfgsParams.epsilon = 0.2;
    double lambdaScale = 100;
    std::cout << "Begin lbfgs\n";
    int totalIters = 0;
    for (int samplingIter = 0; samplingIter < 1; ++samplingIter) {
        if (totalIters >= params.maxIterations)
            break;
        for (; params.smoothing >= params.minSmoothing; params.smoothing /= 2) {
            if (totalIters >= params.maxIterations)
                break;
            std::cout << "\t*** Smoothing: " << params.smoothing << " ***\n";
            auto algData = DeconvolveData<D>{x, b, Q, R, numLambda, constantTerm, lambdaScale, pc, params, stats, primalFn, totalIters};
            lbfgsParams.max_iterations = params.maxIterations - totalIters;
            auto retCode = lbfgs(numDualVars, dualVars.get(), &fVal, deconvolveEvaluate<D>, deconvolveProgress<D>, &algData, &lbfgsParams);
            double primal = primalFn(x) + R.primal(x.data());
            if (primal < -fVal) break;
            std::cout << "\tL-BFGS finished: " << retCode << "\n";
        }
        std::cout << "*** Resampling ***\n";
        R.sampleLabels(x, 1.0/((samplingIter+1)*(samplingIter+1)));
    }

    
    //auto retCode = optimalGradDescent(numDualVars, dualVars.get(), &fVal, deconvolveEvaluate<D>, deconvolveProgress<D>, &algData);

    return x;
}

#define INSTANTIATE_DECONVOLVE(d) \
    template Array<d> Deconvolve<d>(const Array<d>& y, const LinearSystem<d>& H, const LinearSystem<d>& Q, Regularizer<d>& R, ProgressCallback<d>& pc, DeconvolveParams& params, DeconvolveStats& s);
INSTANTIATE_DECONVOLVE(1)
INSTANTIATE_DECONVOLVE(2)
INSTANTIATE_DECONVOLVE(3)
#undef INSTANTIATE_DECONVOLVE
}
