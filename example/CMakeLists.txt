add_executable(image-deconvolve
    "image-deconvolve.cpp"
    "convolve.cpp")

find_package(OpenCV REQUIRED)

target_link_libraries(image-deconvolve
    deconvolve
    boost_program_options
    ${OpenCV_LIBS}
    lbfgs
    fftw3
    )
