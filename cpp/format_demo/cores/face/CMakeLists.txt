cmake_minimum_required(VERSION 3.14)
project(FaceEmbeddingExtractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- 统一库目录 (用户可修改) -----------------
set(PERIMETER_LIBS_DIR /home/nvidia/VSCodeProject/smartboxcore/third_libraries)
message(STATUS "Using centralized library directory: ${PERIMETER_LIBS_DIR}")

# ----------------- OpenCV -----------------
set(OpenCV_DIR ${PERIMETER_LIBS_DIR}/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn calib3d imgcodecs
        PATHS ${OpenCV_DIR}
        NO_DEFAULT_PATH
)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please check your installation or PERIMETER_LIBS_DIR path.")
endif ()

# ----------------- nlohmann/json (header-only) -----------------
# 统一使用 PERIMETER_LIBS_DIR 路径
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
        ${PERIMETER_LIBS_DIR}/json/include
)

# ----------------- 可执行文件 -----------------
add_executable(face_extractor
        main_face_detect.cpp
        FaceAnalyzer.cpp
)

add_executable(face_recognizer
        main_face_recognition.cpp
        FaceAnalyzer.cpp
)

# ----------------- 链接库 -----------------
target_link_libraries(face_extractor PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
)

target_link_libraries(face_recognizer PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
)

# ----------------- 文件系统库 (GCC<9 兼容) -----------------
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(face_extractor PRIVATE stdc++fs)
    target_link_libraries(face_recognizer PRIVATE stdc++fs)
    message(STATUS "Linking against stdc++fs for GCC < 9")
endif ()