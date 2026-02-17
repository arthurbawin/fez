set(TEST_TARGET linear_elasticity)

string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)

deal_ii_pickup_tests()
