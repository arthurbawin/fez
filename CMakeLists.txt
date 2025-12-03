cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======================
# Project options
# ======================
option(ENABLE_TESTING     "Enable testing"                           OFF)
option(FEZ_FORCE_PETSC    "Force PETSc wrappers for linear algebra"  OFF)
option(FEZ_FORCE_TRILINOS "Force Trilinos wrappers for linear algebra" OFF)

# Find deal.II
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR} CONFIG)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

project(fez LANGUAGES CXX)

# ========================================
# Check available linear algebra backends
# ========================================
message(STATUS "---- Linear algebra backend selection ----")

# deal.II provides configuration variables like:
# DEAL_II_WITH_PETSC and DEAL_II_WITH_TRILINOS
set(_dealii_has_petsc     ${DEAL_II_WITH_PETSC})
set(_dealii_has_trilinos  ${DEAL_II_WITH_TRILINOS})

# Determine backend
if(FEZ_FORCE_PETSC AND FEZ_FORCE_TRILINOS)
  message(FATAL_ERROR "You cannot enable both FEZ_FORCE_PETSC and FEZ_FORCE_TRILINOS at the same time.")
endif()

if(FEZ_FORCE_PETSC)
  if(NOT _dealii_has_petsc)
    message(FATAL_ERROR "FEZ_FORCE_PETSC=ON but deal.II was not compiled with PETSc support.")
  endif()
  set(FEZ_WITH_PETSC ON)
  set(FEZ_WITH_TRILINOS OFF)
  message(STATUS "Using PETSc (forced).")

elseif(FEZ_FORCE_TRILINOS)
  if(NOT _dealii_has_trilinos)
    message(FATAL_ERROR "FEZ_FORCE_TRILINOS=ON but deal.II was not compiled with Trilinos support.")
  endif()
  set(FEZ_WITH_TRILINOS ON)
  set(FEZ_WITH_PETSC OFF)
  message(STATUS "Using Trilinos (forced).")

else()
  # Automatic selection: prefer PETSc if available
  if(_dealii_has_petsc)
    set(FEZ_WITH_PETSC ON)
    set(FEZ_WITH_TRILINOS OFF)
    message(STATUS "deal.II supports PETSc; using PETSc backend.")
  elseif(_dealii_has_trilinos)
    set(FEZ_WITH_TRILINOS ON)
    set(FEZ_WITH_PETSC OFF)
    message(STATUS "deal.II supports Trilinos; using Trilinos backend.")
  else()
    message(FATAL_ERROR "Neither PETSc nor Trilinos are available in your deal.II configuration.")
  endif()
endif()

# Define compiler flags
if(FEZ_WITH_PETSC)
  add_compile_definitions(FEZ_WITH_PETSC)
elseif(FEZ_WITH_TRILINOS)
  add_compile_definitions(FEZ_WITH_TRILINOS)
endif()

message(STATUS "FEZ_WITH_PETSC    = ${FEZ_WITH_PETSC}")
message(STATUS "FEZ_WITH_TRILINOS = ${FEZ_WITH_TRILINOS}")
message(STATUS "------------------------------------------")

# === Add Eigen (standard + unsupported) ===
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/contrib/eigen)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR}/unsupported)

# Include the headers
include_directories(${CMAKE_SOURCE_DIR}/include)

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

# Tests
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()