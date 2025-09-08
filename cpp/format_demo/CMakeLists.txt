cmake_minimum_required(VERSION 3.12)
project(det_video_deterministic)

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
endif ()

include_directories(${CUDA_INCLUDE_DIRS})

# ----------------- TensorRT -----------------
# 查找原生 TensorRT 推理所需的库
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu/)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS /usr/lib/aarch64-linux-gnu/)

if (NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "TensorRT libraries ('nvinfer', 'nvonnxparser') not found. Please check your Jetson environment.")
else ()
    message(STATUS "Found TensorRT library: ${NVINFER_LIB}")
    message(STATUS "Found TensorRT ONNX Parser library: ${NVONNXPARSER_LIB}")
endif ()

# 将找到的库放入一个变量中方便使用
set(TRT_LIBS ${NVINFER_LIB} ${NVONNXPARSER_LIB})

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
        demo_cpp.cpp
        feature_processor.cpp
        cores/personReid/PersonReid_dla.cpp
        cores/face/FaceAnalyzer_dla.cpp
)

add_executable(face_extractor
        cores/face/main_face_detect.cpp
        cores/face/FaceAnalyzer.cpp
)

add_executable(face_extractor_dla
        cores/face/main_face_detect_dla.cpp
        cores/face/FaceAnalyzer_dla.cpp
)

add_executable(face_recognizer
        cores/face/main_face_recognition.cpp
        cores/face/FaceAnalyzer.cpp
)

add_executable(face_recognizer_dla
        cores/face/main_face_recognition_dla.cpp
        cores/face/FaceAnalyzer_dla.cpp
)

# 使用 OpenCV DNN 的原始 ReID Demo
add_executable(reid_demo
        cores/personReid/reid_main.cpp
        cores/personReid/PersonReid.cpp
)

# ======================= 【修改的部分在此】 =======================
# 使用 TensorRT DLA 的新 ReID Demo
add_executable(reid_demo_dla
        cores/personReid/reid_main_dla.cpp
        cores/personReid/PersonReid_dla.cpp
)
# ======================= 【修改结束】 =======================

# ----------------- 链接库 -----------------
target_link_libraries(demo_cpp PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        eigen
        ${TRT_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_EXTRA_LIBS}
)

target_link_libraries(face_extractor PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        ${TRT_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_EXTRA_LIBS}
)

target_link_libraries(face_extractor_dla PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        ${TRT_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_EXTRA_LIBS}
)

target_link_libraries(face_recognizer PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        ${TRT_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_EXTRA_LIBS}
)

target_link_libraries(face_recognizer_dla PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        ${TRT_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_EXTRA_LIBS}
)

target_link_libraries(reid_demo PRIVATE
        ${OpenCV_LIBS}
        eigen
)

# ======================= 【修改的部分在此】 =======================
# 为新的 DLA Demo 链接必要的库
target_link_libraries(reid_demo_dla PRIVATE
        ${OpenCV_LIBS}
        eigen
        ${TRT_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_EXTRA_LIBS}
)
# ======================= 【修改结束】 =======================

# ----------------- 文件系统库 -----------------
# 为所有目标统一处理 C++17 filesystem 链接问题
foreach (TARGET demo_cpp face_extractor face_extractor_dla face_recognizer face_recognizer_dla reid_demo reid_demo_dla)
    if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        target_link_libraries(${TARGET} PRIVATE stdc++fs)
    endif ()
endforeach ()

if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message(STATUS "Linking against stdc++fs for GCC < 9")
endif ()