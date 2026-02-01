set(TEST_TARGET incompressible_ns_lambda)

string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)

file(COPY mms_meshes/holed0.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed1.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed2.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed_3d_0.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")
file(COPY mms_meshes/holed_3d_1.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/mms_meshes")

deal_ii_pickup_tests()