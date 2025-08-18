cmake_minimum_required(VERSION 3.14)
project(feature_logic LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

# 指定 Anaconda 环境 Python（例子）
set(Python3_EXECUTABLE "/home/manu/anaconda3/envs/yolo/bin/python")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 从当前 Python 获取 pybind11 包含路径
execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --includes
        OUTPUT_VARIABLE PYBIND11_INCLUDES_STR
)
string(STRIP "${PYBIND11_INCLUDES_STR}" PYBIND11_INCLUDES_STR)
separate_arguments(PYBIND11_INCLUDES_LIST UNIX_COMMAND "${PYBIND11_INCLUDES_STR}")
set(PYBIND11_INCLUDE_DIRS "")
foreach (item ${PYBIND11_INCLUDES_LIST})
    string(REPLACE "-I" "" clean_item ${item})
    list(APPEND PYBIND11_INCLUDE_DIRS ${clean_item})
endforeach ()

add_library(feature_logic MODULE feature_logic.cpp)
target_include_directories(feature_logic PRIVATE ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(feature_logic PRIVATE Python3::Python)
set_target_properties(feature_logic PROPERTIES PREFIX "" SUFFIX ".so")

add_custom_command(TARGET feature_logic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:feature_logic>
        ${PROJECT_SOURCE_DIR}
        COMMENT "复制 feature_logic.so 到源码目录以便 Python import"
)