###
### CMake settings
###
cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0012 NEW)


###
### Project settings
###
project(DECONVOLVE)

set(DECONVOLVE_VERSION_MAJOR "1")
set(DECONVOLVE_VERSION_MINOR "0")
set(DECONVOLVE_VERSION_PATCH "0")
set(DECONVOLVE_VERSION "${DECONVOLVE_VERSION_MAJOR}.${DECONVOLVE_VERSION_MINOR}.${DECONVOLVE_VERSION_PATCH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


###
### Sources, headers, directories and libs
###

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeFiles")

include_directories(${DECONVOLVE_SOURCE_DIR}/include)
include_directories(${DECONVOLVE_SOURCE_DIR}/src)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
include_directories("src/iml")

set(deconvolve-sources
    src/deconvolve.cpp
    src/quadratic-min.cpp
    src/regularizer.cpp
    src/optimal-grad.cpp
    )

###
### Build configuration
### 
#set(CMAKE_BUILD_TYPE Debug)

# GCC settings
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR 
        ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
	#
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG  -ffast-math -g")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
	#
    set(GCC_WARNING_FLAGS "-Wall -Wextra -pedantic -Wold-style-cast -Woverloaded-virtual -Wno-unused-parameter")
    set(GCC_WARNING_FLAGS "${GCC_WARNING_FLAGS} -Werror")
	#
    set(CMAKE_CXX_FLAGS "--std=c++11 ${GCC_WARNING_FLAGS} ${CMAKE_CXX_FLAGS}")
	#
endif()

###
### Target: libdeconvolve
###

add_library(deconvolve STATIC ${deconvolve-sources})


###
### Subdirectories
###

add_subdirectory(test)
add_subdirectory(example)
add_subdirectory(matlab)
