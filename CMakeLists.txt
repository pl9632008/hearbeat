# cmake_minimum_required(VERSION <specify CMake version here>)
project(heart)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)

aux_source_directory(. src)

find_package(OpenCV REQUIRED PATHS /newhome/wangjd/local NO_DEFAULT_PATH )

include_directories(${OpenCV_INCLUDE_DIRS} .)


add_executable(heartbeat ${src})

target_link_libraries(heartbeat ${OpenCV_LIBS})
