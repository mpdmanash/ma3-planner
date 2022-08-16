cmake_minimum_required(VERSION 3.13)
project(ma3_planner)
set(CMAKE_CXX_STANDARD 17)            # Enable c++17 standard

set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
  
find_package(OpenMP)
include_directories(${ma3_planner_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "-O3 -msse3")

FILE(GLOB SOURCE_FILES ${ma3_planner_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}  "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX
)
