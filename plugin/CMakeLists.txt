cmake_minimum_required(VERSION 2.6)

project(mmdeploy_plugins)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /O2")
add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)

find_package(CUDA REQUIRED)

#if(WIN32)
#enable_language(CUDA)
#endif(WIN32)

# cuda
set(cuda_inc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include")
set(cuda_lib "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64")
include_directories(${cuda_inc})
link_directories(${cuda_lib})
#cub
set(CUB_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include/cub")
include_directories(${CUB_ROOT_DIR})
# tensorrt
set(tensorrt_inc "D:/trt_install/TensorRT-8.5.1.7/include")
set(tensorrt_lib "D:/trt_install/TensorRT-8.5.1.7/lib")
include_directories(${tensorrt_inc})
link_directories(${tensorrt_lib})
# opencv
#include_directories("${PROJECT_SOURCE_DIR}/third_party/CV460_64/include")
#set(opencv_lib "${PROJECT_SOURCE_DIR}/third_party/CV460_64/lib/opencv_world460.lib")

# common files
include_directories(common)

file(GLOB grid_sampler_src ${PROJECT_SOURCE_DIR}/grid_sampler/*.cpp ${PROJECT_SOURCE_DIR}/grid_sampler/*.cu)
cuda_add_library(trtgrid_sampler SHARED ${grid_sampler_src})
#cuda_add_library(trtgrid_sampler STATIC ${grid_sampler_src})
target_link_libraries(trtgrid_sampler nvinfer cudart)


#file(GLOB topk_src ${PROJECT_SOURCE_DIR}/gather_topk/*.cpp ${PROJECT_SOURCE_DIR}/gather_topk/*.cu)
#cuda_add_library(trtgather_topk SHARED ${topk_src})
##cuda_add_library(trtgather_topk STATIC ${topk_src})
#target_link_libraries(trtgather_topk nvinfer cudart)


if(UNIX)
    add_definitions(-O2 -pthread)
endif(UNIX)