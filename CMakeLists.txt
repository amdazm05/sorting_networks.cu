cmake_minimum_required(VERSION 3.12...3.14 FATAL_ERROR)
set(CXX_STANDARD 20)
project(cusorts  LANGUAGES CXX)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()
# Good resource for looking into CUDA compilation
# https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9444-build-systems-exploring-modern-cmake-cuda-v2.pdf

include_directories(${CMAKE_SOURCE_DIR}/include)
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/bitonic_sort.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_SOURCE_DIR}/include/bitonic_sort.cuh PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_SOURCE_DIR}/test/bitonic_gpu.cpp PROPERTIES LANGUAGE CXX)

add_library( ${PROJECT_NAME} STATIC CMakeLists.txt
${CMAKE_SOURCE_DIR}/src/bitonic_sort.cu
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

add_executable(bitonic_gpu_test ${CMAKE_SOURCE_DIR}/test/bitonic_gpu.cc)
target_link_libraries(bitonic_gpu_test PUBLIC cuda ${PROJECT_NAME})