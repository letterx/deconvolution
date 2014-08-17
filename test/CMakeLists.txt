set(test-sources
    test-convex-fn.cpp
    test-deconvolve-bp.cpp
    test-project-simplex.cpp
    test-regularizer.cpp
    test-transport.cpp
)

###
### Unit test executable
###

add_executable(unit-test "test-main.cpp" ${test-sources})

target_link_libraries(unit-test deconvolve ${libs} boost_unit_test_framework)

