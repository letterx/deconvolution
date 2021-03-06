#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include "regularizer.hpp"

using namespace deconvolution;

constexpr double epsilon = 0.0001;

BOOST_AUTO_TEST_SUITE(RegularizerTests)

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

    BOOST_AUTO_TEST_CASE(SingleVariable) {
        auto ep = TruncatedL1{0, 0};
        auto R = GridRegularizer<1, TruncatedL1>{std::vector<int>{1}, 2, 1, ep};

        std::vector<double> lambda = { 1.0, 1.0 };
        std::vector<double> gradient = {0.0, 0.0};

        for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
            gradient[0] = 0.0;
            gradient[1] = 0.0;
            auto result = R.evaluate(0, lambda.data(), t, gradient.data(), nullptr);

            BOOST_CHECK_CLOSE(result, -1.0-t*log(2), epsilon);
            BOOST_CHECK_CLOSE(gradient[0], -0.5, epsilon);
            BOOST_CHECK_CLOSE(gradient[1], -0.5, epsilon);
        }

        lambda[1] = 0.0;
        
        for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
            gradient[0] = 0.0;
            gradient[1] = 0.0;
            auto result = R.evaluate(0, lambda.data(), t, gradient.data(), nullptr);

            BOOST_CHECK_CLOSE(result, -1.0-t*log(1+exp(-1.0/t)), epsilon);
            BOOST_CHECK_GE(result, -1.0-t*log(2));
            BOOST_CHECK_CLOSE(gradient[0], -1.0/(1+exp(-1.0/t)), epsilon);
            BOOST_CHECK_CLOSE(gradient[1], -exp(-1.0/t)/(1+exp(-1.0/t)), epsilon);
        }

    }

    BOOST_AUTO_TEST_CASE(MultipleVars) {
        for (int n : {10, 100, 1000}) {
            auto ep = TruncatedL1{0, 0};
            auto R = GridRegularizer<1, TruncatedL1>{std::vector<int>{n}, 2, 1, ep};
            std::vector<double> lambda(n*2, 1.0);
            std::vector<double> gradient(n*2, 0);

            //for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
            double t = 1.0;
            {
                auto result = R.evaluate(0, lambda.data(), t, gradient.data(), nullptr);

                BOOST_CHECK_CLOSE(result, -n - t*n*log(2), epsilon);
                BOOST_CHECK_GE(result, -n - t*n*log(2)-epsilon);
                for (int i = 0; i < 2*n; ++i)
                    BOOST_CHECK_CLOSE(gradient[i], -0.5, epsilon);
            }
        }
    }

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

    BOOST_AUTO_TEST_CASE(Timing) {
        int width = 200;
        int height = 200;
        int nLabels = 32;
        auto ep = TruncatedL1{0, 0};
        auto R = GridRegularizer<2, TruncatedL1>{std::vector<int>{width, height}, nLabels, 1, ep};

        std::vector<double> lambda(width*height*nLabels, 0.0);
        std::vector<double> gradient(width*height*nLabels, 0.0);

        auto startTime = std::chrono::system_clock::now();
        for (int subproblem : {0, 1}) {
            R.evaluate(subproblem, lambda.data(), 1.0, gradient.data(), nullptr);
        }
        double time = std::chrono::duration<double>{std::chrono::system_clock::now() - startTime}.count();
        std::cout << "Time: " << time << "\n";
    }

BOOST_AUTO_TEST_SUITE_END()
