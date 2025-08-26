cmake_minimum_required(VERSION 3.12)
project(det_video_deterministic)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- OpenCV -----------------
find_package(OpenCV REQUIRED)

# 设定你的统一库目录变量，方便管理
set(PERIMETER_LIBS_DIR /home/manu/nfs/perimeter_libs)

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
)

target_link_libraries(demo_cpp PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json
        eigen
)