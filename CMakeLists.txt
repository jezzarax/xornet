#
# xornet

cmake_minimum_required(VERSION 3.27)
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)
set(TEST_TARGET xornet)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
add_subdirectory("vendor/ggml")
add_executable(${TEST_TARGET} "src/main.cpp")
target_link_libraries(${TEST_TARGET} PRIVATE ggml)