# 
# This file is adapted from the CMakeLists.txt of the mmg project.
# It configures some macros and headers, then adds the mmg sources 
# to the main fez library.
#

SET (CMAKE_RELEASE_VERSION_MAJOR "5" )
SET (CMAKE_RELEASE_VERSION_MINOR "8" )
SET (CMAKE_RELEASE_VERSION_PATCH "0" )
SET (CMAKE_RELEASE_DATE "Oct. 30, 2024" )

SET (CMAKE_RELEASE_VERSION
  "${CMAKE_RELEASE_VERSION_MAJOR}.${CMAKE_RELEASE_VERSION_MINOR}.${CMAKE_RELEASE_VERSION_PATCH}")

SET(MMG_DYN_LIB OFF)

# Architecture detection
IF (NOT WIN32)
# Set preprocessor flags to say that we are posix and gnu compatible
  SET (DEF_POSIX "#define MMG_POSIX")
  SET (DEF_GNU "#define MMG_GNU")
ELSEIF(MINGW)
  SET (DEF_GNU "#define MMG_GNU")
ENDIF ()

# Integer type
SET (MMG5_INT "int32_t" CACHE STRING "Integer type")
IF (${MMG5_INT} STREQUAL "int64_t")
  SET ( DEF_MMG5_INT "#define MMG5_int int64_t")
  SET ( DEF_MMG5_INTMAX "#define MMG5_INTMAX LONG_MAX" )
  SET ( DEF_MMG5_PRId "#define MMG5_PRId PRId64")
  SET ( DEF_MMG5F_INT "#define MMG5F_INT SELECTED_INT_KIND(18)")
  SET ( DEF_MMG_SWPBIN "#define MMG5_SWAPBIN(x) MMG5_swapbin_int(x)")
  SET ( DEF_MMG_ABS "#define MMG5_abs(x) (int64_t)labs((int64_t)x)")
ELSE ()
  SET ( DEF_MMG5_INT "#define MMG5_int int32_t")
  SET ( DEF_MMG5_INTMAX "#define MMG5_INTMAX INT_MAX" )
  SET ( DEF_MMG5_PRId "#define MMG5_PRId PRId32")
  SET ( DEF_MMG5F_INT "#define MMG5F_INT SELECTED_INT_KIND(9)")
  SET ( DEF_MMG_SWPBIN "#define MMG5_SWAPBIN(x) MMG5_swapbin(x)")
  SET ( DEF_MMG_ABS "#define MMG5_abs(x) abs(x)")
ENDIF ()

# Create mmgcmakedefines.h file with the architecture infos and the definition
# ot the USE_POINTMAP flag.
SET(MMG_DYN_LIB ${BUILD_SHARED_LIBS} CACHE INTERNAL "" FORCE)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/src/common/mmgcmakedefines.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common/mmgcmakedefines.h @ONLY)

# Create mmgversion.h file with the release infos
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/src/common/mmgversion.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common/mmgversion.h @ONLY)

set(SRC
  # ---------------- COMMON ----------------
  src/common/analys.c
  src/common/anisomovpt.c
  src/common/anisosiz.c
  src/common/API_functions.c
  src/common/API_functionsf.c
  # src/common/apptools.c
  src/common/bezier.c
  src/common/boulep.c
  src/common/chrono.c
  src/common/eigenv.c
  src/common/hash.c
  src/common/inout.c
  src/common/intmet.c
  src/common/isosiz.c
  src/common/librnbg.c
  src/common/libtools.c
  src/common/mettools.c
  src/common/mmg2.c
  src/common/mmg2s.c
  src/common/mmg3.c
  src/common/mmgexterns.c
  src/common/quality.c
  src/common/scalem.c
  src/common/tools.c
  # src/common/vtkparser.cpp // This file requires the VTK library

  # ---------------- MMG2D ----------------
  src/mmg2d/analys_2d.c
  src/mmg2d/anisomovpt_2d.c
  src/mmg2d/anisosiz_2d.c
  src/mmg2d/API_functions_2d.c
  src/mmg2d/API_functionsf_2d.c
  src/mmg2d/bezier_2d.c
  src/mmg2d/boulep_2d.c
  src/mmg2d/cenrad_2d.c
  src/mmg2d/chkmsh_2d.c
  src/mmg2d/colver_2d.c
  src/mmg2d/delone_2d.c
  src/mmg2d/enforcement_2d.c
  src/mmg2d/hash_2d.c
  src/mmg2d/inout_2d.c
  src/mmg2d/inoutcpp_2d.cpp
  src/mmg2d/intmet_2d.c
  src/mmg2d/isosiz_2d.c
  src/mmg2d/length_2d.c
  src/mmg2d/libmmg2d.c
  src/mmg2d/libmmg2df.c
  src/mmg2d/libmmg2d_tools.c
  src/mmg2d/libmmg2d_toolsf.c
  src/mmg2d/lissmet_2d.c
  src/mmg2d/locate_2d.c
  src/mmg2d/mmg2d1.c
  src/mmg2d/mmg2d2.c
  src/mmg2d/mmg2d6.c
  src/mmg2d/mmg2d9.c
  # src/mmg2d/mmg2d.c // Not including file with a main function
  src/mmg2d/mmg2dexterns.c
  src/mmg2d/movpt_2d.c
  src/mmg2d/quality_2d.c
  src/mmg2d/solmap_2d.c
  src/mmg2d/split_2d.c
  src/mmg2d/swapar_2d.c
  src/mmg2d/tools_2d.c
  src/mmg2d/variadic_2d.c
  src/mmg2d/velextls_2d.c
  src/mmg2d/zaldy_2d.c

  # ---------------- MMG3D ----------------
  src/mmg3d/analys_3d.c
  src/mmg3d/anisomovpt_3d.c
  src/mmg3d/anisosiz_3d.c
  src/mmg3d/API_functions_3d.c
  src/mmg3d/API_functionsf_3d.c
  src/mmg3d/bezier_3d.c
  src/mmg3d/boulep_3d.c
  src/mmg3d/cenrad_3d.c
  src/mmg3d/chkmsh_3d.c
  src/mmg3d/colver_3d.c
  src/mmg3d/delaunay_3d.c
  src/mmg3d/hash_3d.c
  src/mmg3d/inout_3d.c
  src/mmg3d/inoutcpp_3d.cpp
  src/mmg3d/intmet_3d.c
  src/mmg3d/isosiz_3d.c
  src/mmg3d/libmmg3d.c
  src/mmg3d/libmmg3df.c
  src/mmg3d/libmmg3d_tools.c
  src/mmg3d/libmmg3d_toolsf.c
  src/mmg3d/librnbg_3d.c
  src/mmg3d/mmg3d1.c
  src/mmg3d/mmg3d1_delone.c
  src/mmg3d/mmg3d1_pattern.c
  src/mmg3d/mmg3d2.c
  src/mmg3d/mmg3d2s.c
  src/mmg3d/mmg3d3.c
  # src/mmg3d/mmg3d.c // Not including file with a main function
  src/mmg3d/mmg3dexterns.c
  src/mmg3d/movpt_3d.c
  src/mmg3d/optbdry_3d.c
  src/mmg3d/optlap_3d.c
  src/mmg3d/opttyp_3d.c
  src/mmg3d/PRoctree_3d.c
  src/mmg3d/quality_3d.c
  src/mmg3d/split_3d.c
  src/mmg3d/swap_3d.c
  src/mmg3d/swapgen_3d.c
  src/mmg3d/tools_3d.c
  src/mmg3d/variadic_3d.c
  src/mmg3d/velextls_3d.c
  src/mmg3d/zaldy_3d.c

  # ---------------- MMGS ----------------
  src/mmgs/analys_s.c
  src/mmgs/anisomovpt_s.c
  src/mmgs/anisosiz_s.c
  src/mmgs/API_functionsf_s.c
  src/mmgs/API_functions_s.c
  src/mmgs/bezier_s.c
  src/mmgs/boulep_s.c
  src/mmgs/chkmsh_s.c
  src/mmgs/colver_s.c
  src/mmgs/gentools_s.c
  src/mmgs/hash_s.c
  src/mmgs/inoutcpp_s.cpp
  src/mmgs/inout_s.c
  src/mmgs/intmet_s.c
  src/mmgs/isosiz_s.c
  src/mmgs/libmmgs.c
  src/mmgs/libmmgsf.c
  src/mmgs/libmmgs_tools.c
  src/mmgs/libmmgs_toolsf.c
  src/mmgs/librnbg_s.c
  src/mmgs/mmgs1.c
  src/mmgs/mmgs2.c
  # src/mmgs/mmgs.c // Not including file with a main function
  src/mmgs/mmgsexterns.c
  src/mmgs/movpt_s.c
  src/mmgs/quality_s.c
  src/mmgs/split_s.c
  src/mmgs/swapar_s.c
  src/mmgs/variadic_s.c
  src/mmgs/zaldy_s.c
)

file(GLOB_RECURSE HDR RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
append_fez_src("${SRC};${HDR}")