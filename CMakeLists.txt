cmake_minimum_required(VERSION 3.15)
project(fez LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find deal.II
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR} CONFIG)

# === Add Eigen (standard + unsupported) ===
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/contrib/eigen)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR}/unsupported)

# Include the headers
include_directories(include)

# === Build a library from src/ ===
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)
add_library(project_src STATIC ${SRC_FILES})
target_include_directories(project_src PUBLIC include)

# Ensure project_src is built with deal.II flags
deal_ii_setup_target(project_src)

# === Build example executables ===
file(GLOB EXAMPLE_FILES CONFIGURE_DEPENDS examples/*.cpp)

foreach(example_src ${EXAMPLE_FILES})
    get_filename_component(example_name ${example_src} NAME_WE)

    add_executable(${example_name} ${example_src})
    target_include_directories(${example_name} PRIVATE include)

    # Link with project source code and deal.II
    target_link_libraries(${example_name} project_src)
    deal_ii_setup_target(${example_name})
endforeach()
