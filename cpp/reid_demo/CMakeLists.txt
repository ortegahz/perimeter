cmake_minimum_required(VERSION 3.14)
project(ReidOnnxDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- 统一库目录 (用户可修改) -----------------
set(PERIMETER_LIBS_DIR /home/nvidia/VSCodeProject/smartboxcore/third_libraries)

# ----------------- CUDA -----------------
find_package(CUDA REQUIRED)

set(CUDA_EXTRA_LIBS "")
if (CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    # 从 CUDA 的lib64目录找 nppim 库路径
    find_library(CUDA_nppicom_LIBRARY nppim
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
            NO_DEFAULT_PATH
    )
    if (CUDA_nppicom_LIBRARY)
        list(APPEND CUDA_EXTRA_LIBS ${CUDA_nppicom_LIBRARY})
        message(STATUS "Found nppim: ${CUDA_nppicom_LIBRARY}")
    else ()
        message(WARNING "CUDA >= 11.0 but libnppim.so not found, skipping link to nppim")
    endif ()
endif ()

include_directories(${CUDA_INCLUDE_DIRS})

# ----------------- Eigen -----------------
set(Eigen3_DIR ${PERIMETER_LIBS_DIR}/eigen/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)

# ----------------- OpenCV -----------------
set(OpenCV_DIR ${PERIMETER_LIBS_DIR}/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please check your installation or PERIMETER_LIBS_DIR path.")
endif ()

# ----------------- 头文件目录 -----------------
include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

# ----------------- 可执行文件 -----------------
add_executable(reid_demo
        reid_main.cpp
        PersonReid.cpp
)

# ----------------- 链接库 -----------------
target_link_libraries(reid_demo PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_npp_LIBRARY}
        ${CUDA_EXTRA_LIBS}  # 这里按条件添加 nppim
        nvinfer
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

# ----------------- 文件系统库 (GCC<9 兼容) -----------------
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(reid_demo PRIVATE stdc++fs)
    message(STATUS "Linking against stdc++fs for GCC < 9")
endif ()

# ----------------- 安装（可选） -----------------
install(TARGETS reid_demo DESTINATION bin)