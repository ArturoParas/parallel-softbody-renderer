cmake_minimum_required(VERSION 3.18)
project(tensileflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

include_directories(./include)
find_library(ASSIMP_LIB assimp)
find_library(GLEW_LIB GLEW)
find_library(GL_LIB GL)
find_library(GLM_LIB glm)
find_library(SFML_GRAPHICS_LIB sfml-graphics)
find_library(SFML_SYSTEM_LIB sfml-system)
find_library(SFML_WINDOW_LIB sfml-window)
find_library(CUDART_LIB cudart)

add_library(
  renderer STATIC
  src/camera.cpp
  src/mesh.cpp
  src/model.cpp
  src/object.cpp
  src/shader.cpp
)
add_library(
  softbody_sim STATIC
  src/cuda_solver.cu
  src/exclusiveScan.cu_inl
)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(
  ${PROJECT_NAME}
  renderer
  softbody_sim
  ${ASSIMP_LIB}
  ${GLEW_LIB}
  ${GL_LIB}
  ${GLM_LIB}
  ${SFML_GRAPHICS_LIB}
  ${SFML_SYSTEM_LIB}
  ${SFML_WINDOW_LIB}
  ${CUDART_LIB}
)
