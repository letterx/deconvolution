#include <boost/test/unit_test.hpp>

#include "transport.hpp"

BOOST_AUTO_TEST_SUITE(TransportTests)

    BOOST_AUTO_TEST_CASE(Example1) {
        std::vector<int> supply = { 5, 3, 8 };
        std::vector<int> demand = { 4, 10, 2 };
        std::vector<int> costs = {
            10, 3, 17,
            5, 12, 4,
            3, 18, 4
        };
        std::vector<int> flow(9);

        solveTransport(3, 3, costs.data(), 
                supply.data(), demand.data(), flow.data());

        std::vector<int> expectedFlow = {
            0, 5, 0,
            0, 3, 0, 
            4, 2, 2
        };

        for (int i = 0; i < 9; ++i)
            BOOST_CHECK_EQUAL(flow[i], expectedFlow[i]);


    }

BOOST_AUTO_TEST_SUITE_END()
