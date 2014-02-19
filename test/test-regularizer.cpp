#include <boost/test/unit_test.hpp>
#include "regularizer.hpp"

using namespace deconvolution;

BOOST_AUTO_TEST_SUITE(RegularizerTests)

    BOOST_AUTO_TEST_CASE(BasicInterface) {
        auto R = GridRegularizer<1>{std::vector<int>{10}, 2, [](int, int) -> double { return 0; }};

        BOOST_CHECK_EQUAL(R.numSubproblems(), 1);
        BOOST_CHECK_EQUAL(R.numLabels(), 2);
        for (int i = 0; i < 10; ++i) {
            BOOST_CHECK_EQUAL(R.getLabel(i, 0), 0);
            BOOST_CHECK_EQUAL(R.getLabel(i, 1), 1);
        }

    }


BOOST_AUTO_TEST_SUITE_END()
