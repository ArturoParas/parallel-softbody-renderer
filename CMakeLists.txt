cmake_minimum_required(VERSION 3.18)
project(tensileflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_library(SFML_GRAPHICS_LIB sfml-graphics)
find_library(SFML_SYSTEM_LIB sfml-system)
find_library(SFML_WINDOW_LIB sfml-window)
find_library(GL_LIB GL)
find_library(GLEW_LIB GLEW)
find_library(GLM_LIB glm)
find_library(ASSIMP_LIB assimp)
find_library(CUDART_LIB cudart)
find_path(CUDA_INCLUDE cuda_runtime.h)

add_library(
  softbody_sim STATIC
  src/cuda_solver.cu
  src/exclusiveScan.cu_inl
)
add_library(
  renderer STATIC
  src/mesh.cpp
  src/shader.cpp
  src/camera.cpp
  src/object.cpp
  src/model.cpp
)

target_include_directories(softbody_sim PUBLIC ./include)
target_include_directories(renderer PUBLIC ./include)

add_executable(${PROJECT_NAME} src/main.cpp)
add_executable(
  create_input
  src/create_input.cpp
  include/create_input.hpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE})

target_link_libraries(
  ${PROJECT_NAME}
  softbody_sim
  renderer
  ${SFML_GRAPHICS_LIB}
  ${SFML_WINDOW_LIB}
  ${SFML_SYSTEM_LIB}
  ${GL_LIB}
  ${GLEW_LIB}
  ${GLM_LIB}
  ${ASSIMP_LIB}
  ${CUDART_LIB}
)
