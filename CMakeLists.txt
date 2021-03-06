cmake_minimum_required(VERSION 3.9)
project(MNIST-NeuralNetwork)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
option(BUILD_SHARED_LIBS "" off)

# add_subdirectory(extern) Not Currently Used

set(SOURCE_FILES
    # MAIN
    source/main.cpp

    # NETWORK
    source/network.cpp

    # MNIST
    source/MNIST/mnist.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_include_directories(${PROJECT_NAME} PUBLIC "extern")
target_include_directories(${PROJECT_NAME} PUBLIC "source") # to allow paths relative to project root

target_link_libraries(${PROJECT_NAME} PRIVATE zlib)
