#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include "deconvolve-bp.hpp"

using namespace deconvolution;

constexpr double epsilon = 0.0001;

BOOST_AUTO_TEST_SUITE(DeconvolveBP)

/* 
 * LambdaOrder Tests
 */

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

/*
 * AddUnaries Tests
 */

    struct DualObjectiveFixture {
        DualObjectiveFixture() {
            b.assign(bVec.begin(), bVec.end());
            lambda[0].assign(l1Vec.begin(), l1Vec.end());
            lambda[1].assign(l2Vec.begin(), l2Vec.end());
        }
        const int nLabels = 2;
        TruncatedL1 ep{1.0, 1.0};
        GridRegularizer<2, TruncatedL1> R { 
            std::vector<int>{2, 3}, nLabels, 1.0, ep
        };
        LinearSystem<2> Q = [](const Array<2>& x) { return x; };
        std::vector<double> bVec = {
            1.0, 0.0,
            -1.0, 2.0 };
        Array<2> b{std::vector<int>{2, 2}};

        Array<2> nu{std::vector<int>{2, 2}};
        
        std::vector<Array<3>> lambda = allocLambda<2>(boost::extents[2][2][2]);
        std::vector<double> l1Vec = {
            0.0, 1.0,   1.0, 1.0,
            2.0, 1.0,  -7.0, 0.0 };
        std::vector<double> l2Vec = {
            0.0, 1.0,   2.0, 1.0,
            1.0, 1.0,  -7.0, 0.0 };
    };


    BOOST_FIXTURE_TEST_CASE(AddUnaries1, DualObjectiveFixture) {
        auto unaries = sumUnaries(R, nu, lambda);
        std::vector<double> expected = {
            0.0, 2.0,   4.0, 2.0, 
            2.0, 2.0,  -14.0, 0.0
        };
        for (int i = 0; i < 8; ++i) {
            if (unaries.data()[i] != expected[i])
                std::cout << "Bad: " << i << "\n";
            BOOST_CHECK_EQUAL(unaries.data()[i], expected[i]);
        }
    }

    BOOST_FIXTURE_TEST_CASE(AddUnaries2, DualObjectiveFixture) {
        std::vector<double> nuVec = {
            -1.0, 0.0,
             1.0,-2.0 };
        nu.assign(nuVec.begin(), nuVec.end());

        auto unaries = sumUnaries(R, nu, lambda);
        std::vector<double> expected = {
            -1.0, 0.0,   4.0, 2.0, 
             2.0, 3.0,  -16.0,-4.0
        };
        for (int i = 0; i < 8; ++i) {
            if (unaries.data()[i] != expected[i])
                std::cout << "Bad: " << i << "\n";
            BOOST_CHECK_EQUAL(unaries.data()[i], expected[i]);
        }
    }

/*
 * DualObjective Tests
 */

    BOOST_FIXTURE_TEST_CASE(DualObjective1, DualObjectiveFixture) {
        double obj = dualObjective(R, Q, b, nu, 0.0, lambda);

        BOOST_CHECK_CLOSE(obj, -19.5, epsilon);
    }

    BOOST_FIXTURE_TEST_CASE(DualObjective2, DualObjectiveFixture) {
        std::vector<double> nuVec = {
            -1.0, 0.0,
             1.0,-2.0 };
        nu.assign(nuVec.begin(), nuVec.end());

        double obj = dualObjective(R, Q, b, nu, 0.0, lambda);

        BOOST_CHECK_CLOSE(obj, 0 + -13.0 + -4.0 + -4.0, epsilon);
    }


/*
 * NuOptimizeLBFGS Tests
 */
    struct NuOptimizeLBFGSFixture {
        LinearSystem<1> Q = [](const Array<1>& x) { return x; };
        Array<1> b { boost::extents[3] };
        std::vector<double> bVec = { 1.0, -2.0, 0.5 };
        TruncatedL1 ep { 1.0, 1.0 };
        GridRegularizer<1, TruncatedL1> R { std::vector<int>{3}, 3, 1.0, ep };
        std::vector<Array<2>> lambda { 1, Array<2>{ boost::extents[3][3] } };
        std::vector<double> lambdaVec = {
            2.0, 0.0, 2.0,
            1.0, 0.5, 0.0,
            0.0, 1.0, 0.0
        };
        double t = 0.1;
        real_1d_array lbfgsX = "[0.0, 0.0, 0.0]";
        real_1d_array lbfgsGrad = "[0.0, 0.0, 0.0]";

        NuOptimizeLBFGSFixture() {
            b.assign(bVec.begin(), bVec.end());
            lambda[0].assign(lambdaVec.begin(), lambdaVec.end());
        }
    };

    BOOST_FIXTURE_TEST_SUITE(ClassNuOptimiseLBFGS, NuOptimizeLBFGSFixture)

        BOOST_AUTO_TEST_CASE(SumLambda) {
            auto sumLambda = NuOptimizeLBFGS<1>::sumLambda(lambda, R);
            BOOST_CHECK_EQUAL(sumLambda[0](0.0), 2.0);
            BOOST_CHECK_EQUAL(sumLambda[0](1.0), 0.0);
            BOOST_CHECK_EQUAL(sumLambda[0](2.0), 0.0);
            BOOST_CHECK_EQUAL(sumLambda[0](3.0), 2.0);

            BOOST_CHECK_EQUAL(sumLambda[1](0.0), 1.0);
            BOOST_CHECK_EQUAL(sumLambda[1](1.0), 0.5);
            BOOST_CHECK_EQUAL(sumLambda[1](2.0), 0.0);
            BOOST_CHECK_EQUAL(sumLambda[1](3.0), 0.0);

            BOOST_CHECK_EQUAL(sumLambda[2](0.0), 0.0);
            BOOST_CHECK_EQUAL(sumLambda[2](1.0), 0.0);
            BOOST_CHECK_EQUAL(sumLambda[2](2.0), 0.0);
            BOOST_CHECK_EQUAL(sumLambda[2](3.0), 0.0);
        }

        BOOST_AUTO_TEST_CASE(Evaluate) {
            auto opt = deconvolution::NuOptimizeLBFGS<1>{ Q, b, R, lambda, t };
            double obj = 0.0;
            opt._evaluate(lbfgsX, obj, lbfgsGrad);
            BOOST_CHECK_CLOSE(obj, 0 + 1.8 + .9875 + 0, epsilon);
            BOOST_CHECK_CLOSE(lbfgsGrad[0], -3.0, epsilon);
            BOOST_CHECK_CLOSE(lbfgsGrad[1],  1.5, epsilon);
            BOOST_CHECK_CLOSE(lbfgsGrad[2], -0.5, epsilon);

        }


    BOOST_AUTO_TEST_SUITE_END()
    
BOOST_AUTO_TEST_SUITE_END()
