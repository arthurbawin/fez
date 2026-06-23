cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project options
option(ENABLE_TESTING        "Enable testing"                             OFF)
option(FEZ_WITH_MMG          "Enable MMG mesh adaptation library"         OFF)
option(FEZ_WITH_BUNDLED_MMG  "Use bundled version of the MMG library"     OFF)

# For this option, see the comments for the function create_hp_line_dof_identities()
# in incompressible_ns_solver_lambda.h.
# By default, we specify that deal.II has the hp identities bug.
# This adds some computation time for the hp solvers.
option(DEAL_II_WITH_HP_IDENTITIES_BUG "" ON)

# Lagrange multiplier equations do not typically need a source term,
# so there is none for speed. To be able to test with arbitrary manufactured
# solutions, however, a source term is required and this option must be enabled.
option(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM "" OFF)

# deal.II added a dimension template to ReferenceCell on the development branch,
# and since this is in between a version release, we cannot check the deal.II
# version to discriminate.
option(REFERENCE_CELL_WITH_TEMPLATE_PARAMETER "" OFF)

# Find deal.II
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR} CONFIG)
deal_ii_initialize_cached_variables()

# Project name, specify both C++ and C languages to compile bundled
# libraries in C (e.g., mmg).
project(fez CXX C)

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Check that deal.II was compiled with PETSc.
# This is required for now, as it is the only linear algebra
# library that is interfaced.
message(STATUS)
message(STATUS "Detecting PETSc")
if (${DEAL_II_WITH_PETSC})
  set(FEZ_WITH_PETSC ON)
  add_compile_definitions(FEZ_WITH_PETSC)
  message(STATUS "\tPETSc sucessfully set up from deal.II")
else()
  message(FATAL_ERROR "\tfez requires deal.II to be compiled with PETSc support")
endif()

if(DEAL_II_WITH_HP_IDENTITIES_BUG)
  add_compile_definitions(DEAL_II_WITH_HP_LINE_IDENTITIES_BUG)
endif()

if(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
  add_compile_definitions(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
endif()

if(REFERENCE_CELL_WITH_TEMPLATE_PARAMETER)
  add_compile_definitions(REFERENCE_CELL_WITH_TEMPLATE_PARAMETER)
endif()

# Macro to add a bundled library to the project sources
# Barely adapted from Gmsh's CMakeLists.txt
macro(append_fez_src FILES)
  foreach(FILE ${FILES})
    file(REAL_PATH ${FILE} ABSOLUTE_PATH BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    list(APPEND LIST ${ABSOLUTE_PATH})
  endforeach()
  set(CONTRIB_SRC_FILES ${CONTRIB_SRC_FILES};${LIST} PARENT_SCOPE)
endmacro()

# Eigen (standard + unsupported)
message(STATUS)
message(STATUS "Detecting Eigen")
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/contrib/eigen)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR}/unsupported)
message(STATUS "\tEigen sucessfully set up from contrib/eigen")

# Mmg
if(FEZ_WITH_MMG)
  message(STATUS)
  message(STATUS "Detecting MMG...")
  if(FEZ_WITH_BUNDLED_MMG)
    message(STATUS "\tUsing bundled version of MMG")
    add_subdirectory(contrib/mmg_v5_8_0)
    include_directories(contrib/mmg_v5_8_0)
    include_directories(contrib/mmg_v5_8_0/src/common)
    message(STATUS "\tMMG successfully set up from contrib/mmg_v5_8_0")
  else()
    message(STATUS "\tUsing system version of MMG")
    find_package(MMG REQUIRED)
    message(STATUS "\tMMG_FOUND:        ${MMG_FOUND}")
    message(STATUS "\tMMG_LIBRARY_DIRS: ${MMG_LIBRARY_DIRS}")
    message(STATUS "\tMMG_LIBRARIES:    ${MMG_LIBRARIES}")
    message(STATUS "\tMMG_INCLUDE_DIRS: ${MMG_INCLUDE_DIRS}")
    list(APPEND EXTERNAL_INCLUDES ${MMG_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${MMG_LIBRARIES})
    message(STATUS "\n\tMMG successfully set up with external dependencies.")
  endif()
  add_compile_definitions(FEZ_WITH_MMG)
endif()

# Options summary
message(STATUS)
message(STATUS "Summary of the configuration:")
message(STATUS "\tFEZ_WITH_PETSC                         = ${FEZ_WITH_PETSC}")
message(STATUS "\tFEZ_WITH_MMG                           = ${FEZ_WITH_MMG}")
message(STATUS "\tFEZ_WITH_BUNDLED_MMG                   = ${FEZ_WITH_BUNDLED_MMG}")
message(STATUS "\tDEAL_II_WITH_HP_IDENTITIES_BUG         = ${DEAL_II_WITH_HP_IDENTITIES_BUG}")
message(STATUS "\tLAGRANGE_MULTIPLIER_WITH_SOURCE_TERM   = ${LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM}")
message(STATUS "\tREFERENCE_CELL_WITH_TEMPLATE_PARAMETER = ${REFERENCE_CELL_WITH_TEMPLATE_PARAMETER}")
message(STATUS "\tENABLE_TESTING                         = ${ENABLE_TESTING}")
message(STATUS)

# fez headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create library:
# Get fez source files
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)

# Append the sources from contrib/ to the project sources
set(SRC_FILES ${SRC_FILES};${CONTRIB_SRC_FILES})

add_library(fez STATIC ${SRC_FILES})
target_include_directories(fez PUBLIC include PUBLIC ${EXTERNAL_INCLUDES})
target_link_libraries(fez ${EXTERNAL_LIBRARIES})

# Link with deal.II and add deal.II compile flags
deal_ii_setup_target(fez)

# === Build solvers executables ===
file(GLOB SOLVER_FILES CONFIGURE_DEPENDS solvers/*.cpp)

foreach(solver_src ${SOLVER_FILES})
    get_filename_component(solver_name ${solver_src} NAME_WE)

    add_executable(${solver_name} ${solver_src})
    target_include_directories(${solver_name} PRIVATE include)

    # Link with project source code and deal.II
    target_link_libraries(${solver_name} fez)
    deal_ii_setup_target(${solver_name})
endforeach()

# Tests
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()