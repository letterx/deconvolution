add_executable(image-deconvolve
    "image-deconvolve.cpp"
    "convolve.cpp"
    "psnr.cpp")

find_package(OpenCV REQUIRED)

target_link_libraries(image-deconvolve
    deconvolve
    boost_program_options
    ${libs}
    ${OpenCV_LIBS}
    lbfgs
    fftw3
    )

add_executable(psnr
    "psnr-main.cpp"
    "psnr.cpp")

target_link_libraries(psnr ${OpenCV_LIBS} boost_program_options)
