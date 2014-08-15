#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include "deconvolve-bp.hpp"

using namespace deconvolution;

constexpr double epsilon = 0.0001;

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



    BOOST_AUTO_TEST_CASE(DualObjective) {
        const int nLabels = 2;
        auto ep = deconvolution::TruncatedL1{1.0, 1.0};
        auto R = deconvolution::GridRegularizer<2, 
             deconvolution::TruncatedL1>{
            std::vector<int>{2, 3}, 
            nLabels, 1.0, ep
        };
        LinearSystem<2> Q = [](const Array<2>& x) { return x; };
        std::vector<double> bVec = {
            1.0, 0.0,
            -1.0, 2.0 };
        Array<2> b{std::vector<int>{2, 2}};
        b.assign(bVec.begin(), bVec.end());

        Array<2> nu{std::vector<int>{2, 2}};
        
        auto lambda = allocLambda<2>(boost::extents[2][2][2]);
        std::vector<double> l1Vec = {
            0.0, 1.0,   1.0, 1.0,
            2.0, 1.0,   0.0, 0.0 };
        std::vector<double> l2Vec = {
            0.0, 0.0,   0.0, 0.0,
            0.0, 0.0,   0.0, 0.0 };
        lambda[0].assign(l1Vec.begin(), l1Vec.end());
        lambda[1].assign(l2Vec.begin(), l2Vec.end());

        double obj = dualObjective(R, Q, b, nu, 0.0, lambda);

        BOOST_CHECK_CLOSE(obj, -3.5, epsilon);
    }

BOOST_AUTO_TEST_SUITE_END()
