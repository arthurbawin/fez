file(COPY meshes/starting_mesh.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY meshes/rectangular_box.msh DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(heat_adapt)
