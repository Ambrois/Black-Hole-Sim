cmake_minimum_required(VERSION 3.16)
project(blackholesim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Main raytracing app
add_executable(blackhole src/raytrace.cpp)
target_link_libraries(blackhole PRIVATE sfml-graphics sfml-window sfml-system)
