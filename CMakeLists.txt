cmake_minimum_required(VERSION 3.16)
project(blackholesim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Library
add_library(physics STATIC src/rk4.cpp)
target_include_directories(physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Main app
add_executable(bh_cpu app/bh_cpu.cpp)
target_link_libraries(bh_cpu PRIVATE physics sfml-graphics sfml-window sfml-system)

# Test target
add_executable(test_rk4 tests/test_rk4.cpp)
target_link_libraries(test_rk4 PRIVATE physics)

