file(COPY meshes/starting_mesh.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY meshes/rectangular_box.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY meshes/rising_bubble.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

if (DEAL_II_WITH_LAPACK)
	add_subdirectory(heat_adapt)
	add_subdirectory(incompressible_chns_adapt)
endif()
