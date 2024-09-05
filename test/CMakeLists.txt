include(../cmake/cuda.cmake)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)

aux_source_directory(../test DIR_TEST)
aux_source_directory(../test/test_base DIR_TEST_BASE)

add_executable(test_llm ${DIR_TEST_BASE})

target_link_libraries(test_llm ${link_ext_lib} ${CUDA_LIBRARIES})

target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)