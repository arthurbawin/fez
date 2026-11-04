set(TEST_TARGET monolithic_fsi)

string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)

file(COPY cylinder_2d.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY cylinder_2d_coarse.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY cylinder_2d_45_degrees.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY cylinder_3d.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY mms_meshes/holed0.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed1.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed_with_one_physical_per_boundary0.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed_with_one_physical_per_boundary1.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")

deal_ii_pickup_tests()


# The comparison reports mismatched entries without throwing. Fail explicitly
# on that diagnostic, rather than comparing floating-point output to a golden file.
set(_rotation_jacobian_workdir
    "${CMAKE_CURRENT_BINARY_DIR}/jacobian_matrix_artificial_rotation/run")
file(MAKE_DIRECTORY "${_rotation_jacobian_workdir}")
add_test(NAME monolithic_fsi/artificial_rotation_jacobian
  COMMAND $<TARGET_FILE:monolithic_fsi>
    "${CMAKE_CURRENT_SOURCE_DIR}/jacobian_matrix_artificial_rotation.prm")
set_tests_properties(monolithic_fsi/artificial_rotation_jacobian PROPERTIES
  WORKING_DIRECTORY "${_rotation_jacobian_workdir}"
  FAIL_REGULAR_EXPRESSION "Max abs error on elem;Exception;Aborting;[Nn][Aa][Nn];[Ii][Nn][Ff]inity"
  TIMEOUT 180)
