cmake_minimum_required(VERSION 3.10)

project(vision_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 若未指定构建类型，默认 Release（对视频处理性能影响明显）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "Ceres version:  ${Ceres_VERSION}")

# 所有可执行文件生成到 build/ 根目录，便于统一调用
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ----------------------------------------------------------------------------
# 公共头文件目录（放通用工具，如后续扩展）
# ----------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# ----------------------------------------------------------------------------
# 任务 1：OpenCV 图片处理
#   仅依赖 OpenCV，职责单一
# ----------------------------------------------------------------------------
add_executable(task1_image
    src/task1_image/main.cpp
)
target_include_directories(task1_image PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task1_image PRIVATE ${OpenCV_LIBS})

# ----------------------------------------------------------------------------
# 任务 2：合成旋转视频的参数拟合
# 依赖 OpenCV（读写视频/图像处理）+ Eigen3 + Ceres（非线性优化）
#----------------------------------------------------------------------------
add_executable(task2_fit
    src/task2_fit/main.cpp
)
target_include_directories(task2_fit PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task2_fit PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
)

