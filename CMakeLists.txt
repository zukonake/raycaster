cmake_minimum_required(VERSION 2.8)
project(raycaster)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wundef")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Og -pipe")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM "include")


file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(raycaster ${SOURCES})

find_library(GLFW_LIB glfw)
find_library(GL_LIB GL)
find_library(DL_LIB dl)

target_link_libraries(raycaster "${GLFW_LIB}")
target_link_libraries(raycaster "${GL_LIB}")
target_link_libraries(raycaster "${DL_LIB}")

