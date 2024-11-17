cmake_minimum_required(VERSION 3.18)
project(tensileflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(./include)

add_library(
  renderer STATIC
  # src/
)
add_library(
  softbody_sim STATIC
  # src/
)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(
  ${PROJECT_NAME}
  renderer
  softbody_sim
)
