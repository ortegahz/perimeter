# CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(ReidOnnxDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------- 统一库目录 (用户可修改) -----------------
# 设定您的统一库目录变量，方便管理
# 如果OpenCV不在标准路径，请取消注释并修改此路径
# set(PERIMETER_LIBS_DIR /home/manu/nfs/perimeter_libs)
# find_package(OpenCV REQUIRED PATHS ${PERIMETER_LIBS_DIR}/opencv/build)

# ----------------- OpenCV -----------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn imgcodecs)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please check your installation or PERIMETER_LIBS_DIR path.")
endif ()

# ----------------- 头文件目录 -----------------
# 允许编译器找到我们项目中的头文件 (例如 include/PersonReid.h)
include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

# ----------------- 可执行文件 -----------------
# 定义可执行文件及其源文件
add_executable(reid_demo
        reid_main.cpp
        PersonReid.cpp
)

# ----------------- 链接库 -----------------
# 将 OpenCV 库链接到可执行文件
target_link_libraries(reid_demo PRIVATE
        ${OpenCV_LIBS}
)

# ----------------- 文件系统库 (可选但建议保留) -----------------
# 为兼容旧版GCC（如GCC 8），显式链接filesystem库
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(reid_demo PRIVATE stdc++fs)
    message(STATUS "Linking against stdc++fs for GCC < 9")
endif ()

# ----------------- 安装（可选） -----------------
install(TARGETS reid_demo DESTINATION bin)