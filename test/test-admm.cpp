#include <boost/test/unit_test.hpp>

#include "admm.hpp"

#include <iostream>

using namespace deconvolution;

BOOST_AUTO_TEST_SUITE(AdmmTests)
    BOOST_AUTO_TEST_CASE(Step1) {
        auto R = GridRangeRegularizer<1>{std::vector<int>{2}, 2, 1, 1, 1.0, 1.0}; 
        
        DeconvolveParams params;
        real_1d_array hessianDiag;
        hessianDiag.setlength(16);

        minlbfgsstate lbfgsState;

        double mu1[] = {
            0.0, 1.0, 1.0, 0.0,
        };
        double nu[] = {
            0, 0, 0, 0,
        };

        auto admm = AdmmRegularizerLbfgs<1>{2, &R, &params, &hessianDiag, 
            &lbfgsState, mu1, nu};

        double lambda[] = {
            0, 0, 0, 0,
        };
        double grad[] = {
            0, 0, 0, 0,
        };

        // Checking unaries
        double obj = 0;
        for (int i = 0; i < 2; ++i) 
            obj += admm._evaluateUnary(i, lambda, grad, hessianDiag.getcontent());

        {
            double expectedGrad[] = {
                0.0, 1.0, 1.0, 0.0
            };

            for (int idx = 0; idx < 4; ++idx) {
                BOOST_CHECK_CLOSE(grad[idx], expectedGrad[idx], 1.0e-6);
            }
            BOOST_CHECK_CLOSE(obj, 0.0, 1.0e-6);
        }

        // Checking regularizer
        obj += R.evaluate(0, lambda, 0.01, grad, hessianDiag.getcontent());

        {
            double expectedGrad[] = {
                -0.5, 0.5, 0.5, -0.5
            };

            for (int idx = 0; idx < 4; ++idx) {
                BOOST_CHECK_CLOSE(grad[idx], expectedGrad[idx], 1.0e-6);
            }
            BOOST_CHECK_CLOSE(obj, -0.01*log(2), 1.0e-6);
        }

    }

    BOOST_AUTO_TEST_CASE(Step2) {
        auto R = GridRangeRegularizer<1>{std::vector<int>{2}, 2, 1, 1, 1.0, 1.0}; 
        
        DeconvolveParams params;
        real_1d_array hessianDiag;
        hessianDiag.setlength(16);

        minlbfgsstate lbfgsState;

        double mu1[] = {
            0.0, 1.0, 1.0, 0.0,
        };
        double nu[] = {
            0, 0, 0, 0,
        };

        auto admm = AdmmRegularizerLbfgs<1>{2, &R, &params, &hessianDiag, 
            &lbfgsState, mu1, nu};

        double lambda[] = {
            -0.5, 0.5, 0.5, -0.5,
        };
        double grad[] = {
            0, 0, 0, 0,
        };

        // Checking unaries
        double obj = 0;

        {
            obj += admm._evaluateUnary(0, lambda, grad, hessianDiag.getcontent());
            BOOST_CHECK_CLOSE(obj, 0.5, 1.0e-6);
            obj += admm._evaluateUnary(1, lambda, grad, hessianDiag.getcontent());
            BOOST_CHECK_CLOSE(obj, 1.0, 1.0e-6);

            double expectedGrad[] = {
                0.0, 1.0, 1.0, 0.0
            };

            for (int idx = 0; idx < 4; ++idx) {
                BOOST_CHECK_CLOSE(grad[idx], expectedGrad[idx], 1.0e-6);
            }
            BOOST_CHECK_CLOSE(obj, 1.0, 1.0e-6);
        }

        // Checking regularizer
        const double t = 0.0001;
        obj += R.evaluate(0, lambda, t, grad, hessianDiag.getcontent());

        {
            const double third = 0.333333333333333333;
            double expectedGrad[] = {
                -third, third, third, -third
            };

            for (int idx = 0; idx < 4; ++idx) {
                BOOST_CHECK_CLOSE(grad[idx], expectedGrad[idx], 1.0e-6);
            }
            BOOST_CHECK_CLOSE(obj, 1.0-t*log(3+exp(-2.0/t)), 1.0e-6);
        }

    }

    BOOST_AUTO_TEST_CASE(FullRun) {
        auto R = GridRangeRegularizer<1>{std::vector<int>{2}, 2, 1, 1, 1.0, 1.0}; 
        
        DeconvolveParams params;
        real_1d_array hessianDiag;
        hessianDiag.setlength(16);

        minlbfgsstate lbfgsState;

        double mu1[] = {
            0.0, 1.0, 1.0, 0.0,
        };
        double nu[] = {
            0, 0, 0, 0,
        };

        auto admm = AdmmRegularizerLbfgs<1>{2, &R, &params, &hessianDiag, 
            &lbfgsState, mu1, nu};

        double lambda[] = {
            0, 0, 0, 0,
        };

        real_1d_array lbfgsX;
        lbfgsX.setcontent(4, lambda);

        minlbfgsreport lbfgsReport;

        minlbfgscreate(1, lbfgsX, lbfgsState);
        minlbfgssetxrep(lbfgsState, true);

        minlbfgssetcond(lbfgsState, 0.001, 0.0, 0, params.maxIterations);
        minlbfgsrestartfrom(lbfgsState, lbfgsX);
        minlbfgsoptimize(lbfgsState, AdmmRegularizerLbfgs<1>::evaluate,
                AdmmRegularizerLbfgs<1>::progress, &admm);
        minlbfgsresults(lbfgsState, lbfgsX, lbfgsReport);
        BOOST_CHECK_EQUAL(lbfgsReport.terminationtype, 4);

    }

BOOST_AUTO_TEST_SUITE_END()
