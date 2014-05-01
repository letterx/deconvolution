#include <boost/test/unit_test.hpp>

#include "project-simplex.hpp"

void checkProject(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> result;
    projectSimplex(x, result);
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        BOOST_CHECK_CLOSE(result[i], y[i], 0.000001);
    }
}

BOOST_AUTO_TEST_SUITE(ProjectSimplexTests)

    BOOST_AUTO_TEST_CASE(Examples) {
        checkProject({0.25, 0.25, 0.25, 0.25}, {0.25, 0.25, 0.25, 0.25});
        checkProject({0.0, 0.0, 0.0, 0.0}, {0.25, 0.25, 0.25, 0.25});
        checkProject({-100.0, -100.0, -100.0, -100.0}, {0.25, 0.25, 0.25, 0.25});
        checkProject({100.0, 100.0, 100.0, 100.0}, {0.25, 0.25, 0.25, 0.25});

        checkProject({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
        checkProject({0.0, 1.0, 0.0}, {0.0, 1.0, 0.0});
        checkProject({0.0, 0.0, 1.0}, {0.0, 0.0, 1.0});

        checkProject({2.0, -0.4, -0.6}, {1.0, 0.0, 0.0});
        checkProject({2.0, -0.6, -0.4}, {1.0, 0.0, 0.0});
        checkProject({-0.6, 2.0, -0.4}, {0.0, 1.0, 0.0});
        checkProject({-0.4, 2.0, -0.6}, {0.0, 1.0, 0.0});
        checkProject({-0.4, -0.6, 2.0}, {0.0, 0.0, 1.0});
        checkProject({-0.6, -0.4, 2.0}, {0.0, 0.0, 1.0});

        checkProject({-0.4, 0.6, 0.8}, {0.0, 0.4, 0.6});
        checkProject({-0.4, 0.8, 0.6}, {0.0, 0.6, 0.4});
        checkProject({0.6, -0.4, 0.8}, {0.4, 0.0, 0.6});
        checkProject({0.8, -0.4, 0.6}, {0.6, 0.0, 0.4});
        checkProject({0.8, 0.6, -0.4}, {0.6, 0.4, 0.0});
        checkProject({0.6, 0.8, -0.4}, {0.4, 0.6, 0.0});


    }

BOOST_AUTO_TEST_SUITE_END()
