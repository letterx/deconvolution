#include <boost/test/unit_test.hpp>
#include "regularizer.hpp"

using namespace deconvolution;

constexpr double epsilon = 0.0001;

BOOST_AUTO_TEST_SUITE(RegularizerTests)

    BOOST_AUTO_TEST_CASE(BasicInterface) {
        auto R = GridRegularizer<1>{std::vector<int>{10}, 2, 1, [](int, int) -> double { return 0; }};

        BOOST_CHECK_EQUAL(R.numSubproblems(), 1);
        BOOST_CHECK_EQUAL(R.numLabels(), 2);
        for (int i = 0; i < 10; ++i) {
            BOOST_CHECK_EQUAL(R.getLabel(i, 0), 0);
            BOOST_CHECK_EQUAL(R.getLabel(i, 1), 1);
        }
    }

    BOOST_AUTO_TEST_CASE(SingleVariable) {
        auto R = GridRegularizer<1>{std::vector<int>{1}, 2, 1, [](int, int) -> double { return 0; }};

        std::vector<double> lambda = { 1.0, 1.0 };
        std::vector<double> gradient = {0.0, 0.0};

        for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
            auto result = R.evaluate(0, lambda.data(), t, gradient.data());

            BOOST_CHECK_CLOSE(result, -1.0-t*log(2), epsilon);
            BOOST_CHECK_CLOSE(gradient[0], -0.5, epsilon);
            BOOST_CHECK_CLOSE(gradient[1], -0.5, epsilon);
        }

        lambda[1] = 0.0;
        
        for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
            auto result = R.evaluate(0, lambda.data(), t, gradient.data());

            BOOST_CHECK_CLOSE(result, -1.0-t*log(1+exp(-1.0/t)), epsilon);
            BOOST_CHECK_GE(result, -1.0-t*log(2));
            BOOST_CHECK_CLOSE(gradient[0], -1.0/(1+exp(-1.0/t)), epsilon);
            BOOST_CHECK_CLOSE(gradient[1], -exp(-1.0/t)/(1+exp(-1.0/t)), epsilon);
        }

    }

    BOOST_AUTO_TEST_CASE(MultipleVars) {
        for (int n : {10, 100, 1000}) {
            auto R = GridRegularizer<1>(std::vector<int>{n}, 2, 1, 
                    [](int l1, int l2) -> double {
                        return 0;
                    });
            std::vector<double> lambda(n*2, 1.0);
            std::vector<double> gradient(n*2, 0);

            //for (double t = 1024.0; t >= 1.0/1024.0; t /= 2) {
            double t = 1.0;
            {
                auto result = R.evaluate(0, lambda.data(), t, gradient.data());

                BOOST_CHECK_CLOSE(result, -n - t*n*log(2), epsilon);
                BOOST_CHECK_GE(result, -n - t*n*log(2)-epsilon);
                for (int i = 0; i < 2*n; ++i)
                    BOOST_CHECK_CLOSE(gradient[i], -0.5, epsilon);
            }
        }
    }

BOOST_AUTO_TEST_SUITE_END()
