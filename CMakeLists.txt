cmake_minimum_required(VERSION 3.16)
project(shape_classifier C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 生成 compile_commands.json,供 clangd / LSP 读取正确的 include 路径,消除编辑器里的误报
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 用系统自带的 OpenMP 让训练跑满多核
find_package(OpenMP REQUIRED)

add_executable(shape_classifier
    src/main.c
    src/layer.c
    src/network.c
    src/dataset.c
    src/rng.c
)

target_include_directories(shape_classifier PRIVATE include)

# 说明: gcc 最高优化档是 -O3(-O4 会被当作 -O3 处理,不会更强)。
# 这里显式用 -O3,并加 -march=native 让编译器针对本机指令集优化。
target_compile_options(shape_classifier PRIVATE
    -O3
    -Wall -Wextra
    -march=native
)

if(OpenMP_C_FOUND)
    target_link_libraries(shape_classifier PRIVATE OpenMP::OpenMP_C)
endif()

if(UNIX)
    target_link_libraries(shape_classifier PRIVATE m)
endif()
