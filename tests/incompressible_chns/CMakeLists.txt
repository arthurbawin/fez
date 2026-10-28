set(TEST_TARGET incompressible_chns)

string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)

deal_ii_pickup_tests()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(_pressure_gauge_mpi_args)
  set(_pressure_gauge_processes 1)
  if(DEAL_II_MPIEXEC)
    set(_pressure_gauge_mpi_args --mpiexec ${DEAL_II_MPIEXEC})
    set(_pressure_gauge_processes 2)
  endif()
  add_test(
    NAME incompressible_chns/quasi_pressure_gauge
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_quasi_pressure_gauge.py
      --solver $<TARGET_FILE:incompressible_chns>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/quasi_pressure_gauge
      ${_pressure_gauge_mpi_args})
  set_tests_properties(incompressible_chns/quasi_pressure_gauge
    PROPERTIES TIMEOUT 360 PROCESSORS ${_pressure_gauge_processes}
      FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  add_test(
    NAME incompressible_chns/chns_postprocessing
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_chns_postprocessing.py
      --solver $<TARGET_FILE:incompressible_chns>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/chns_postprocessing)
  set_tests_properties(incompressible_chns/chns_postprocessing
    PROPERTIES TIMEOUT 600 PROCESSORS 1
      FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  add_test(
    NAME incompressible_chns/stepien_postprocessing
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_stepien_postprocessing.py
      --solver $<TARGET_FILE:incompressible_chns>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/stepien_postprocessing)
  set_tests_properties(incompressible_chns/stepien_postprocessing
    PROPERTIES TIMEOUT 600 PROCESSORS 1
      FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  add_test(
    NAME incompressible_chns/chns_phase_energy
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_chns_phase_energy.py
      --solver $<TARGET_FILE:incompressible_chns>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/chns_phase_energy)
  set_tests_properties(incompressible_chns/chns_phase_energy
    PROPERTIES TIMEOUT 600 PROCESSORS 1
      FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  add_test(
    NAME incompressible_chns/quasi_incompressible_invariants
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_quasi_incompressible.py
      --solver $<TARGET_FILE:incompressible_chns>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/quasi_incompressible_invariants
      --suite core)
  set_tests_properties(incompressible_chns/quasi_incompressible_invariants
    PROPERTIES TIMEOUT 600 PROCESSORS 1
      FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  add_test(
    NAME incompressible_chns/quasi_incompressible_physics
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_quasi_incompressible.py
      --solver $<TARGET_FILE:incompressible_chns>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/quasi_incompressible_physics
      --suite physics)
  set_tests_properties(incompressible_chns/quasi_incompressible_physics
    PROPERTIES TIMEOUT 600 PROCESSORS 1
      FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  add_test(
    NAME incompressible_chns/quasi_incompressible_bdf2
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_quasi_incompressible.py
      --solver $<TARGET_FILE:incompressible_chns>
      --workdir ${CMAKE_CURRENT_BINARY_DIR}/quasi_incompressible_bdf2
      --suite bdf2)
  set_tests_properties(incompressible_chns/quasi_incompressible_bdf2
    PROPERTIES TIMEOUT 600 PROCESSORS 1
      FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  if(DEAL_II_MPIEXEC)
    add_test(
      NAME incompressible_chns/chns_postprocessing_mpi
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/validate_chns_postprocessing.py
        --solver $<TARGET_FILE:incompressible_chns>
        --workdir ${CMAKE_CURRENT_BINARY_DIR}/chns_postprocessing_mpi
        --mpiexec ${DEAL_II_MPIEXEC})
    set_tests_properties(incompressible_chns/chns_postprocessing_mpi
      PROPERTIES TIMEOUT 600 PROCESSORS 2
        FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
    add_test(
      NAME incompressible_chns/stepien_postprocessing_mpi
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/validate_stepien_postprocessing.py
        --solver $<TARGET_FILE:incompressible_chns>
        --workdir ${CMAKE_CURRENT_BINARY_DIR}/stepien_postprocessing_mpi
        --mpiexec ${DEAL_II_MPIEXEC})
    set_tests_properties(incompressible_chns/stepien_postprocessing_mpi
      PROPERTIES TIMEOUT 600 PROCESSORS 2
        FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
    add_test(
      NAME incompressible_chns/chns_phase_energy_mpi
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/validate_chns_phase_energy.py
        --solver $<TARGET_FILE:incompressible_chns>
        --workdir ${CMAKE_CURRENT_BINARY_DIR}/chns_phase_energy_mpi
        --mpiexec ${DEAL_II_MPIEXEC})
    set_tests_properties(incompressible_chns/chns_phase_energy_mpi
      PROPERTIES TIMEOUT 600 PROCESSORS 2
        FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
    add_test(
      NAME incompressible_chns/quasi_incompressible_mpi
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/validate_quasi_incompressible.py
        --solver $<TARGET_FILE:incompressible_chns>
        --workdir ${CMAKE_CURRENT_BINARY_DIR}/quasi_incompressible_mpi
        --mpiexec ${DEAL_II_MPIEXEC})
    set_tests_properties(incompressible_chns/quasi_incompressible_mpi
      PROPERTIES TIMEOUT 600 PROCESSORS 2
        FIXTURES_REQUIRED "test_dependency/incompressible_chns.executable")
  endif()
endif()
