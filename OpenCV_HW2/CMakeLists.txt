cmake_minimum_required(VERSION 3.6)
project(OpenCV_HW2)

find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(OpenCV_HW2 ${SOURCE_FILES})
target_link_libraries( OpenCV_HW2 ${OpenCV_LIBS} )