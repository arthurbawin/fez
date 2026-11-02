# Mesh adaptation tests for the incompressible NS solver
set(TEST_TARGET incompressible_ns)
string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)
deal_ii_pickup_tests()
