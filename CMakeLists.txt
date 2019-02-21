cmake_minimum_required (VERSION 2.6)
project (yolov3-tiny)

set (CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} src/main.cpp)