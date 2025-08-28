cmake_minimum_required(VERSION 3.14)
project(FaceEmbeddingExtractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- 统一库目录 -----------------
# 设定你的统一库目录变量，方便管理
set(PERIMETER_LIBS_DIR /home/manu/nfs/perimeter_libs)
message(STATUS "Using centralized library directory: ${PERIMETER_LIBS_DIR}")

# ----------------- OpenCV -----------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# ----------------- nlohmann/json (header-only) -----------------
# 假设 json.hpp 的路径是: ${PERIMETER_LIBS_DIR}/json/include/json.hpp
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
        ${PERIMETER_LIBS_DIR}/json/include
)

# ----------------- 可执行文件 -----------------
# 定义可执行文件及其源文件
# 【修改】为每个 main 文件分别创建一个可执行文件
add_executable(face_extractor
        main_face_detect.cpp
        FaceAnalyzer.cpp
)

add_executable(face_recognizer
        main_face_recognition.cpp
        FaceAnalyzer.cpp
)

# ----------------- 链接库 -----------------
# 【修改】将库链接到两个可执行文件
target_link_libraries(face_extractor PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
)

target_link_libraries(face_recognizer PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
)

# ----------------- 文件系统库（可选但建议保留） -----------------
# 为兼容旧版GCC（如GCC 8），显式链接filesystem库
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    # 【修改】为两个可执行文件都链接 stdc++fs
    target_link_libraries(face_extractor face_recognizer PRIVATE stdc++fs)
endif ()