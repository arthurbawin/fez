set(TEST_TARGET monolithic_fsi)

string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)

file(COPY cylinder_2d.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY mms_meshes/holed0.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed1.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed_with_one_physical_per_boundary0.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed_with_one_physical_per_boundary1.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")

deal_ii_pickup_tests()