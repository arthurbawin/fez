set(TEST_TARGET incompressible_chns_ale)

string(TOLOWER ${CMAKE_BUILD_TYPE} _build_type)

deal_ii_pickup_tests()

# Keep the other ALE tests available with older deal.II versions.
if(NOT DEAL_II_WITH_MAPPING_FE_FIELD_HESSIANS)
  get_property(_chns_ale_tests DIRECTORY PROPERTY TESTS)
  foreach(_test IN LISTS _chns_ale_tests)
    if(_test MATCHES "^incompressible_chns_ale/(jacobian_[23]d_stabilized|mms_2d_bdf2_spacetime_stabilized)\\.")
      set_tests_properties("${_test}" PROPERTIES DISABLED TRUE)
    endif()
  endforeach()
endif()
