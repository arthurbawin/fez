file(COPY meshes/starting_mesh.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

if (DEAL_II_WITH_LAPACK)
	add_subdirectory(heat_adapt)
endif()
