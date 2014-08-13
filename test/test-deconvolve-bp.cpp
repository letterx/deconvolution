#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include "deconvolve-bp.hpp"

using namespace deconvolution;

//constexpr double epsilon = 0.0001;

BOOST_AUTO_TEST_SUITE(DeconvolveBP)

    BOOST_AUTO_TEST_CASE(LambdaOrder) {
        {
            auto order = lambdaOrder<1>(0);
            auto shape = boost::extents[20][16];
            auto array = Array<2>{shape, order};

            BOOST_CHECK_EQUAL(array.strides()[0], 16);
            BOOST_CHECK_EQUAL(array.strides()[1], 1);
        }

        {
            auto order = lambdaOrder<2>(0);
            auto shape = boost::extents[20][13][16];
            auto array = Array<3>{shape, order};

            BOOST_CHECK_EQUAL(array.strides()[0], 16);
            BOOST_CHECK_EQUAL(array.strides()[1], 16*20);
            BOOST_CHECK_EQUAL(array.strides()[2], 1);
        }
        {
            auto order = lambdaOrder<2>(1);
            auto shape = boost::extents[20][13][16];
            auto array = Array<3>{shape, order};

            BOOST_CHECK_EQUAL(array.strides()[0], 16*13);
            BOOST_CHECK_EQUAL(array.strides()[1], 16);
            BOOST_CHECK_EQUAL(array.strides()[2], 1);
        }

        {
            auto order = lambdaOrder<3>(0);
            auto shape = boost::extents[7][20][13][16];
            auto array = Array<4>{shape, order};

            BOOST_CHECK_EQUAL(array.strides()[0], 16);
            BOOST_CHECK_EQUAL(array.strides()[3], 1);
        }
        {
            auto order = lambdaOrder<3>(1);
            auto shape = boost::extents[7][20][13][16];
            auto array = Array<4>{shape, order};

            BOOST_CHECK_EQUAL(array.strides()[1], 16);
            BOOST_CHECK_EQUAL(array.strides()[3], 1);
        }
        {
            auto order = lambdaOrder<3>(2);
            auto shape = boost::extents[7][20][13][16];
            auto array = Array<4>{shape, order};

            BOOST_CHECK_EQUAL(array.strides()[2], 16);
            BOOST_CHECK_EQUAL(array.strides()[3], 1);
        }

    }

BOOST_AUTO_TEST_SUITE_END()
