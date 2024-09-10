include(../cmake/cuda.cmake)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(CUDA REQUIRED)  # 确保找到 CUDA

set(link_ext_lib glog::glog GTest::gtest)

# 添加 CUDA 库的搜索
list(APPEND link_ext_lib ${CUDA_LIBRARIES} cuda cudart)
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT /usr/local/cuda)

aux_source_directory(../test DIR_TEST)
aux_source_directory(../test/test_base DIR_TEST_BASE)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_BASE} ${DIR_TEST_TENSOR})

target_link_libraries(test_llm ${link_ext_lib} ${CUDA_LIBRARIES})

target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ../include)

target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)