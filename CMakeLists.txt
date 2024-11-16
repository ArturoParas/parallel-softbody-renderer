cmake_minimum_required(3.18)
project(tensileflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(
  SOURCES
  src/main.cpp
)
set(
  HEADERS
  # include/...
)

add_executable(
  ${PROJECT_NAME}
  ${SOURCES}
  ${HEADERS}
)
