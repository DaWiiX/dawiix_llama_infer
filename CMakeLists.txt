cmake_minimum_required(VERSION 3.16)

# ---- export compile commands ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Set the CUDA compiler path or name ----
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# ---- set project name and language ----
project(llama_infer CXX CUDA)

# ---- import customized cmake modules ----
include(cmake/cuda.cmake)

# ---- set c++ cuda standard ----
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" OFF)
if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)
endif()

# ---- find dependencies ----
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# ---- Add source files ----
aux_source_directory(src/base/ DIR_BASE)

# ---- set library output directory ----
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ---- add library ----
add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})

# ---- link output library to dependencies ----
target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

# ---- set include directories ----
target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ---- set compile definitions ----
if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

# ---- cuda separable compilation ----
set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- add subdirectories cmake ----
add_subdirectory(test)
# add_subdirectory(demo)