cmake_minimum_required(VERSION 3.12)
project(det_video_deterministic)

# ======================= 【修改的部分在此】 =======================
# 添加此行来设置策略，告诉CMake继续使用旧的FindCUDA模块，以消除警告
cmake_policy(SET CMP0146 OLD)
# ======================= 【修改结束】 =======================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- OpenCV -----------------
find_package(OpenCV REQUIRED)

# 设定你的统一库目录变量，方便管理
set(PERIMETER_LIBS_DIR /home/manu/nfs/perimeter_libs)

# ----------------- CUDA (参考 ReidOnnxDemo) -----------------
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
        message(STATUS "Found nppim for linking: ${CUDA_nppicom_LIBRARY}")
    else ()
        message(WARNING "CUDA >= 11.0 but libnppim.so not found, skipping link.")
    endif ()
endif ()

# 包含 CUDA 头文件目录，以便 OpenCV DNN 后端能找到相关头文件
include_directories(${CUDA_INCLUDE_DIRS})

# ----------------- nlohmann/json (header-only) -----------------
# 假设：/home/manu/nfs/perimeter_libs/json/include/json.hpp
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
        ${PERIMETER_LIBS_DIR}/json/include
)

# ----------------- Eigen (header-only) -----------------
# 假设：/home/manu/nfs/perimeter_libs/eigen/Eigen/...
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE
        ${PERIMETER_LIBS_DIR}/eigen
)

# ----------------- 可执行文件 -----------------
add_executable(demo_cpp
        demo_cpp.cpp
        feature_processor.cpp
        # 假设这些文件位于 cores/ 目录下
        cores/personReid/PersonReid.cpp
        cores/face/FaceAnalyzer.cpp
)

target_link_libraries(demo_cpp PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        eigen
        # 添加 CUDA 库链接
        ${CUDA_LIBRARIES}
        ${CUDA_npp_LIBRARY}
        ${CUDA_EXTRA_LIBS}
)

# ----------------- 文件系统库 (GCC<9 兼容) -----------------
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(demo_cpp PRIVATE stdc++fs)
endif ()