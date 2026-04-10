set(TEST_LIBRARIES project_src)
set(TEST_TARGET unit_tests)

file(COPY meshes/cylinder.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

deal_ii_pickup_tests()
