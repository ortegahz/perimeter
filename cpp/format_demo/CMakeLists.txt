cmake_minimum_required(VERSION 3.12)
project(det_video_deterministic)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- OpenCV -----------------
find_package(OpenCV REQUIRED)

# ----------------- nlohmann/json (header-only) -----------------
#   libs/json/include 里面有 json.hpp
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/json/include
)

# ----------------- Eigen (header-only) -----------------
#   libs/eigen 里面有 Eigen/Core Eigen/Dense ...
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen
)

# ----------------- 可执行文件 -----------------
add_executable(demo_cpp
        demo_cpp.cpp
        feature_processor.cpp
)

target_link_libraries(demo_cpp PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        eigen
)