set(test-sources
)

###
### Unit test executable
###

add_executable(unit-test "test-main.cpp" ${test-sources})

target_link_libraries(unit-test deconvolve ${libs} boost_unit_test_framework)

