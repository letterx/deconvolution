add_custom_target(deconvolveMex ALL DEPENDS deconvolve
    COMMAND mex -I${CMAKE_SOURCE_DIR}/include -I/usr/local/include ${CMAKE_SOURCE_DIR}/matlab/deconvolveDual.cpp -L${CMAKE_BINARY_DIR} -ldeconvolve -llbfgs -ltbb)  

