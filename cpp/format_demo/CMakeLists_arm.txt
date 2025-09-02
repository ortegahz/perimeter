cmake_minimum_required(VERSION 3.12)
project(det_video_deterministic)

# ======================= 【已修复】 =======================
# 删除了旧版CMake不兼容的 cmake_policy(SET CMP0146 OLD) 这一行
# ======================= 【修复结束】 =======================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PERIMETER_LIBS_DIR /home/nvidia/VSCodeProject/smartboxcore/third_libraries)
message(STATUS "Using centralized third-party library directory: ${PERIMETER_LIBS_DIR}")

set(OpenCV_DIR ${PERIMETER_LIBS_DIR}/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED
        PATHS ${OpenCV_DIR}
        NO_DEFAULT_PATH
)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Found OpenCV version for ARM: ${OpenCV_VERSION}")
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found in the specified PERIMETER_LIBS_DIR. Please check the path and your ARM build of OpenCV.")
endif ()

# ----------------- CUDA -----------------
find_package(CUDA REQUIRED)

set(CUDA_EXTRA_LIBS "")
if (CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    # ======================= 【已修复】 =======================
    # 修正了变量名的拼写错误：CUDA_nppicom_LIBRARY -> CUDA_nppim_LIBRARY
    find_library(CUDA_nppim_LIBRARY nppim
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
            NO_DEFAULT_PATH
    )
    if (CUDA_nppim_LIBRARY)
        list(APPEND CUDA_EXTRA_LIBS ${CUDA_nppim_LIBRARY})
        message(STATUS "Found nppim for linking: ${CUDA_nppim_LIBRARY}")
    else ()
        message(WARNING "CUDA >= 11.0 but libnppim.so not found, skipping link.")
    endif ()
    # ======================= 【修复结束】 =======================
endif ()

include_directories(${CUDA_INCLUDE_DIRS})

# ----------------- nlohmann/json (header-only) -----------------
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
        ${PERIMETER_LIBS_DIR}/json/include
)

# ----------------- Eigen (header-only) -----------------
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE
        ${PERIMETER_LIBS_DIR}/eigen
)

# ----------------- 可执行文件 -----------------
add_executable(demo_cpp
        demo_cpp.cpp # 请确保这是您的主文件名
        feature_processor.cpp
        cores/personReid/PersonReid.cpp
        cores/face/FaceAnalyzer.cpp
)

# ----------------- 链接库 -----------------
# ======================= 【已修复】 =======================
# 移除了不稳定的 ${CUDA_npp_LIBRARY}。我们手动找到的库已包含在 ${CUDA_EXTRA_LIBS} 中
target_link_libraries(demo_cpp PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        eigen
        ${CUDA_LIBRARIES}
        ${CUDA_EXTRA_LIBS}
)
# ======================= 【修复结束】 =======================

# ----------------- 文件系统库 -----------------
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(demo_cpp PRIVATE stdc++fs)
    message(STATUS "Linking against stdc++fs for GCC < 9")
endif ()