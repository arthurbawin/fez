# Reuse the historical parameter files and meshes; keep generated inputs/logs
# in the build tree. LU tests validate FGMRES independently of PC quality.
find_package(Python3 COMPONENTS Interpreter QUIET)
find_program(FSI_MPIEXEC NAMES mpiexec mpirun)
if(Python3_Interpreter_FOUND AND FSI_MPIEXEC)
  foreach(case cylinder_2d_with_mass_and_damping fsi_2d_bdf2_coupled_variable_lame jacobian_matrix)
    foreach(ranks 1 4)
      add_test(NAME iterative_fsi/${case}.mpirun=${ranks}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare.py
          --solver $<TARGET_FILE:iterative_fsi>
          --reference $<TARGET_FILE:monolithic_fsi>
          --source ${CMAKE_SOURCE_DIR}
          --work ${CMAKE_CURRENT_BINARY_DIR}/${case}-${ranks}
          --case ${case} --ranks ${ranks} --mpiexec ${FSI_MPIEXEC})
      set_tests_properties(iterative_fsi/${case}.mpirun=${ranks}
        PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "fsi;iterative;reference")
    endforeach()
  endforeach()
else()
  message(STATUS "iterative_fsi regression tests need Python3 and mpiexec")
endif()
