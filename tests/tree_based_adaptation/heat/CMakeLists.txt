# Mesh adaptation tests for the heat equation solver
set(TEST_TARGET heat)
string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)
deal_ii_pickup_tests()
