#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include <limits>
#include "convex-fn.hpp"

constexpr double epsilon = 0.0000001;

BOOST_AUTO_TEST_SUITE(TestConvexFn)

    struct ConvexFnFixture {
        std::vector<double> xVals = { -1.0, 0.0, 1.0, 2.0, 3.0 };
        std::vector<double> fVals = { 2.0, 4.0, -1.0, 3.0, 4.0 };
        PiecewiseLinearFn pl { xVals.begin(), xVals.end(), fVals.begin() };
        ConvexFn f = pl.convexify();
    };

    BOOST_FIXTURE_TEST_SUITE(ExampleFn, ConvexFnFixture) 

        BOOST_AUTO_TEST_CASE(PiecewiseLinearVals) {
            BOOST_CHECK_EQUAL(pl(-1.1), std::numeric_limits<double>::max());
            BOOST_CHECK_CLOSE(pl(-1.0), 2.0, epsilon);
            BOOST_CHECK_CLOSE(pl(-0.9), 2.2, epsilon);
            BOOST_CHECK_CLOSE(pl(-0.1), 3.8, epsilon);
            BOOST_CHECK_CLOSE(pl(0.0), 4.0, epsilon);
            BOOST_CHECK_CLOSE(pl(0.1), 3.5, epsilon);
            BOOST_CHECK_CLOSE(pl(0.9),-0.5, epsilon);
            BOOST_CHECK_CLOSE(pl(1.0),-1.0, epsilon);
            BOOST_CHECK_CLOSE(pl(1.1),-0.6, epsilon);
            BOOST_CHECK_CLOSE(pl(1.9), 2.6, epsilon);
            BOOST_CHECK_CLOSE(pl(2.0), 3.0, epsilon);
            BOOST_CHECK_CLOSE(pl(2.1), 3.1, epsilon);
            BOOST_CHECK_CLOSE(pl(2.9), 3.9, epsilon);
            BOOST_CHECK_CLOSE(pl(3.0), 4.0, epsilon);
            BOOST_CHECK_EQUAL(pl(3.1), std::numeric_limits<double>::max());
        }

        BOOST_AUTO_TEST_CASE(ConvexVals) {
            BOOST_CHECK_EQUAL(f(-1.1), std::numeric_limits<double>::max());
            BOOST_CHECK_CLOSE(f(-1.0), 2.0, epsilon);
            BOOST_CHECK_CLOSE(f(-0.9), 1.85, epsilon);
            BOOST_CHECK_CLOSE(f(0.0), 0.5, epsilon);
            BOOST_CHECK_CLOSE(f(0.9),-0.85, epsilon);
            BOOST_CHECK_CLOSE(f(1.0),-1.0, epsilon);
            BOOST_CHECK_CLOSE(f(1.1),-0.75, epsilon);
            BOOST_CHECK_CLOSE(f(2.0), 1.5, epsilon);
            BOOST_CHECK_CLOSE(f(2.9), 3.75, epsilon);
            BOOST_CHECK_CLOSE(f(3.0), 4.0, epsilon);
            BOOST_CHECK_EQUAL(f(3.1), std::numeric_limits<double>::max());
        }

        BOOST_AUTO_TEST_CASE(Moreau) {
            const double t = 0.1;

            BOOST_CHECK_CLOSE(f.moreauEnvelope(-2.00, t), 7.0   , epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope(-1.16, t), 2.128 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope(-1.15, t), 2.1125, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope(-1.00, t), 1.8875, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 0.00, t), 0.3875, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 0.85, t),-0.8875, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 1.00, t),-1.0   , epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 1.25, t),-0.6875, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 2.00, t), 1.1875, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 3.00, t), 3.6875, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 3.25, t), 4.3125, epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 3.26, t), 4.338 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauEnvelope( 4.00, t), 9.0   , epsilon);
        }

        BOOST_AUTO_TEST_CASE(MoreauGrad) {
            const double t = 0.1;

            BOOST_CHECK_CLOSE(f.moreauGrad(-2.00, t),-10.0, epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad(-1.16, t),-1.6 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad(-1.15, t),-1.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad(-1.00, t),-1.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 0.00, t),-1.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 0.85, t),-1.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 1.00, t), 0.0 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 1.25, t), 2.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 2.00, t), 2.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 3.00, t), 2.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 3.25, t), 2.5 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 3.26, t), 2.6 , epsilon);
            BOOST_CHECK_CLOSE(f.moreauGrad( 4.00, t), 10.0, epsilon);
        }

    BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE_END()
