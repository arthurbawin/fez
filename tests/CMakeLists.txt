# MappingFEField Hessian transformations require deal.II PR #20215, which
# is not available in the 9.7/9.8 releases. Enable these tests only with a
# deal.II build containing that correction.
option(DEAL_II_WITH_MAPPING_FE_FIELD_HESSIANS
       "Run tests requiring MappingFEField Hessian transformations" OFF)

add_subdirectory(compressible_ns)
add_subdirectory(incompressible_ns)
add_subdirectory(incompressible_ns_lambda)
add_subdirectory(monolithic_fsi)
add_subdirectory(fsi)
add_subdirectory(incompressible_chns)
add_subdirectory(incompressible_chns_ale)
add_subdirectory(heat)
add_subdirectory(linear_elasticity)
add_subdirectory(unit_tests)
add_subdirectory(unit_tests/metric_tensor)
add_subdirectory(unit_tests/metric_field)
add_subdirectory(unit_tests/stabilization)

if (DEAL_II_WITH_LAPACK)
	add_subdirectory(unit_tests/transient_fixed_point)
endif()

if(FEZ_WITH_MMG)
	add_subdirectory(metric_based_adaptation)
endif()

if(DEAL_II_WITH_P4EST)
	add_subdirectory(tree_based_adaptation)
endif()
