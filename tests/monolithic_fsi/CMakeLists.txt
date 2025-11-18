set(TEST_TARGET monolithic_fsi)

string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)

file(COPY cylinder_2d.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

deal_ii_pickup_tests()