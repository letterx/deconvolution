#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include "regularizer.hpp"

using namespace deconvolution;

constexpr double epsilon = 0.0001;

BOOST_AUTO_TEST_SUITE(RegularizerHPP)

    BOOST_AUTO_TEST_SUITE(ClassTruncatedL1)

        BOOST_AUTO_TEST_CASE(EdgeFn) {
            auto ep = TruncatedL1{10.0, 7.0};

            BOOST_CHECK_CLOSE(ep.edgeFn(0.0, 0.0), 0.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(1.0, 1.0), 0.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(-210.0, -210.0), 0.0, epsilon);

            BOOST_CHECK_CLOSE(ep.edgeFn(1.0, 2.0), 7.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(2.0, 1.0), 7.0, epsilon);

            BOOST_CHECK_CLOSE(ep.edgeFn(1.0, 7.0), 42.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(-5.0, 5.0), 70.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(-5.0, 6.0), 70.0, epsilon);
        }

        BOOST_AUTO_TEST_CASE(EdgeGrad) {
            auto ep = TruncatedL1{10.0, 7.0};

            double g1, g2;
            BOOST_CHECK_THROW(
                    ep.edgeGrad(0.0, 0.0, g1, g2), 
                    DeconvolutionAssertion);
        }

    BOOST_AUTO_TEST_SUITE_END()

    BOOST_AUTO_TEST_SUITE(ClassSmoothEdge)

        BOOST_AUTO_TEST_CASE(EdgeFn) {
            auto ep = SmoothEdge{7.0, 10.0};

            BOOST_CHECK_CLOSE(ep.edgeFn(0.0, 0.0), 0.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(1.0, 1.0), 0.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(-210.0, -210.0), 0.0, epsilon);

            BOOST_CHECK_CLOSE(ep.edgeFn(1.0, 2.0), 7.0/11.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(2.0, 1.0), 7.0/11.0, epsilon);

            BOOST_CHECK_CLOSE(ep.edgeFn(1.0, 7.0), 7.0*36.0/46.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(-5.0, 5.0), 7.0*100.0/110.0, epsilon);
            BOOST_CHECK_CLOSE(ep.edgeFn(-5.0, 6.0), 7.0*121.0/131.0, epsilon);
        }

        BOOST_AUTO_TEST_CASE(EdgeGrad) {
            auto ep = SmoothEdge{7.0, 10.0};

            {
                double g1 = 0;
                double g2 = 0;
                ep.edgeGrad(0.0, 0.0, g1, g2);
                BOOST_CHECK_SMALL(g1, epsilon);
                BOOST_CHECK_SMALL(g2, epsilon);
            }

            {
                double g1 = 0;
                double g2 = 0;
                ep.edgeGrad(3.0, 4.0, g1, g2);
                BOOST_CHECK_CLOSE(g1,-2.0*7.0*10.0/121.0, epsilon);
                BOOST_CHECK_CLOSE(g2, 2.0*7.0*10.0/121.0, epsilon);
            }

            {
                double g1 = 0;
                double g2 = 0;
                ep.edgeGrad(5.0, 4.0, g1, g2);
                BOOST_CHECK_CLOSE(g1, 2.0*7.0*10.0/121.0, epsilon);
                BOOST_CHECK_CLOSE(g2,-2.0*7.0*10.0/121.0, epsilon);
            }

        }

    BOOST_AUTO_TEST_SUITE_END()


    BOOST_AUTO_TEST_CASE(BasicInterface) {
        auto ep = TruncatedL1{0, 0};
        auto R = GridRegularizer<1, TruncatedL1>{std::vector<int>{10}, 2, 1, ep};

        BOOST_CHECK_EQUAL(R.numSubproblems(), 1);
        BOOST_CHECK_EQUAL(R.numLabels(), 2);
        for (int i = 0; i < 10; ++i) {
            BOOST_CHECK_EQUAL(R.getLabel(i, 0), 0);
            BOOST_CHECK_EQUAL(R.getLabel(i, 1), 1);
        }
    }

/*
 *    BOOST_AUTO_TEST_CASE(SingleVariable) {
 *        auto ep = TruncatedL1{0, 0};
 *        auto R = GridRegularizer<1, TruncatedL1>{std::vector<int>{1}, 2, 1, ep};
 *
 *        std::vector<double> lambda = { 1.0, 1.0 };
 *        std::vector<double> gradient = {0.0, 0.0};
 *
 *        for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
 *            gradient[0] = 0.0;
 *            gradient[1] = 0.0;
 *            auto result = R.evaluate(0, lambda.data(), t, gradient.data(), nullptr);
 *
 *            BOOST_CHECK_CLOSE(result, -1.0-t*log(2), epsilon);
 *            BOOST_CHECK_CLOSE(gradient[0], -0.5, epsilon);
 *            BOOST_CHECK_CLOSE(gradient[1], -0.5, epsilon);
 *        }
 *
 *        lambda[1] = 0.0;
 *        
 *        for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
 *            gradient[0] = 0.0;
 *            gradient[1] = 0.0;
 *            auto result = R.evaluate(0, lambda.data(), t, gradient.data(), nullptr);
 *
 *            BOOST_CHECK_CLOSE(result, -1.0-t*log(1+exp(-1.0/t)), epsilon);
 *            BOOST_CHECK_GE(result, -1.0-t*log(2));
 *            BOOST_CHECK_CLOSE(gradient[0], -1.0/(1+exp(-1.0/t)), epsilon);
 *            BOOST_CHECK_CLOSE(gradient[1], -exp(-1.0/t)/(1+exp(-1.0/t)), epsilon);
 *        }
 *
 *    }
 *
 *    BOOST_AUTO_TEST_CASE(MultipleVars) {
 *        for (int n : {10, 100, 1000}) {
 *            auto ep = TruncatedL1{0, 0};
 *            auto R = GridRegularizer<1, TruncatedL1>{std::vector<int>{n}, 2, 1, ep};
 *            std::vector<double> lambda(n*2, 1.0);
 *            std::vector<double> gradient(n*2, 0);
 *
 *            //for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
 *            double t = 1.0;
 *            {
 *                auto result = R.evaluate(0, lambda.data(), t, gradient.data(), nullptr);
 *
 *                BOOST_CHECK_CLOSE(result, -n - t*n*log(2), epsilon);
 *                BOOST_CHECK_GE(result, -n - t*n*log(2)-epsilon);
 *                for (int i = 0; i < 2*n; ++i)
 *                    BOOST_CHECK_CLOSE(gradient[i], -0.5, epsilon);
 *            }
 *        }
 *    }
 */

    BOOST_AUTO_TEST_CASE(Primal) {
        auto ep = TruncatedL1{3.0, 1.0};
        auto R = GridRegularizer<2, TruncatedL1>{std::vector<int>{3, 4}, 2, 1, ep};

        double x[] = {
            4.0, 1.0, 1.0, 7.0,
            3.0, 7.0, 7.0, 5.0,
            0.0,-2.0, 2.0, 4.0
        };

        auto primal = R.primal(x, nullptr);
        BOOST_CHECK_CLOSE(primal, 18 + 19, epsilon);
    }


BOOST_AUTO_TEST_SUITE_END()
