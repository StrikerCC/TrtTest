cmake_minimum_required(VERSION 3.10)
project(TrtTest)

set(CMAKE_CXX_STANDARD 17)


find_package(Threads)
set(CUDAToolkit_ROOT C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/bin)
#set(CUDA_TOOLKIT_ROOT_DIR C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/bin)
#set(CUDA_INCLUDE_DIRS C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/include)
#set(CUDA_CUDART_LIBRARY C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/lib/x64)
#find_package(CUDA REQUIRED)
find_package(CUDAToolkit  REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_INCLUDE_DIRS" ${CUDA_INCLUDE_DIRS})

set(OpenCV_DIR D:/source/opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(TRT_DIR C:/Users/cheng.chen05/Downloads/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6)
#set(TensorRT_INCLUDE_DIRS C:/Users/cheng.chen05/Downloads/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/include)
#set(TensorRT_LIBRARIES C:/Users/cheng.chen05/Downloads/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib)
#find_package(TensorRT REQUIRED)
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)
#link_directories(C:/Users/cheng.chen05/Downloads/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib)

add_executable(TrtTest
        main.cpp
        nv_test.hpp
        preprocess.cpp)

target_include_directories(TrtTest PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
#target_link_libraries(TrtTest PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
#target_link_libraries(TrtTest PRIVATE CUDA::cublas)

target_link_libraries(TrtTest PRIVATE nvinfer nvinfer_plugin nvparsers nvonnxparser)
#target_link_libraries(TrtTest TRT_LIBRARIES)
target_link_libraries(TrtTest PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(TrtTest PRIVATE ${OpenCV_LIBS})

############################################

# TARGETS --------------------------------------------------------------------------------------------------------------
#add_executable(trt_sample trt_sample.cpp)
#target_include_directories(trt_sample PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
#target_link_libraries(trt_sample PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
