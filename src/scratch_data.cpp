
#include <scratch_data.h>

template <int dim>
ScratchDataNS<dim>::ScratchDataNS(const FESystem<dim>        &fe,
                                  const Quadrature<dim>      &cell_quadrature,
                                  const Mapping<dim>         &mapping,
                                  const Quadrature<dim - 1>  &face_quadrature,
                                  const unsigned int          dofs_per_cell,
                                  const std::vector<double>  &bdfCoeffs,
                                  const ParameterReader<dim> &param)
  : fe_values(mapping, fe, cell_quadrature, required_updates)
  , fe_face_values(mapping, fe, face_quadrature, required_face_updates)
  , n_q_points(cell_quadrature.size())

  // We assume simplicial meshes with all tris or tets
  , n_faces((dim == 3) ? 4 : 3)

  , n_faces_q_points(face_quadrature.size())
  , dofs_per_cell(dofs_per_cell)
  , bdfCoeffs(bdfCoeffs)
{
  velocity.first_vector_component = u_lower;
  pressure.component              = p_lower;
  this->allocate();

  // Check if weak forms are to be assembled on boundaries
  has_boundary_forms = false;
  for (const auto &[id, bc] : param.fluid_bc)
  {
    if (bc.type == BoundaryConditions::Type::open_mms)
    {
      has_boundary_forms = true;
      break;
    }
  }
}

template <int dim>
ScratchDataNS<dim>::ScratchDataNS(const ScratchDataNS &other)
  : has_boundary_forms(other.has_boundary_forms)
  , fe_values(other.fe_values.get_mapping(),
              other.fe_values.get_fe(),
              other.fe_values.get_quadrature(),
              required_updates)
  , fe_face_values(other.fe_face_values.get_mapping(),
                   other.fe_face_values.get_fe(),
                   other.fe_face_values.get_quadrature(),
                   required_face_updates)
  , n_q_points(other.n_q_points)
  , n_faces(other.n_faces)
  , n_faces_q_points(other.n_faces_q_points)
  , dofs_per_cell(other.dofs_per_cell)
  , bdfCoeffs(other.bdfCoeffs)
{
  velocity.first_vector_component = u_lower;
  pressure.component              = p_lower;
  this->allocate();
}

template <int dim>
void ScratchDataNS<dim>::allocate()
{
  JxW.resize(n_q_points);
  components.resize(dofs_per_cell);

  present_velocity_values.resize(n_q_points);
  present_velocity_gradients.resize(n_q_points);
  present_pressure_values.resize(n_q_points);
  // BDF
  previous_velocity_values.resize(bdfCoeffs.size() - 1,
                                  std::vector<Tensor<1, dim>>(n_q_points));

  source_term_full.resize(n_q_points, Vector<double>(n_components));
  source_term_velocity.resize(n_q_points);
  source_term_pressure.resize(n_q_points);

  grad_source_term_full.resize(n_q_points,
                               std::vector<Tensor<1, dim>>(n_components));
  grad_source_velocity.resize(n_q_points);
  grad_source_pressure.resize(n_q_points);

  phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
  grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
  div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
  phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));

  // Faces
  face_boundary_id.resize(n_faces);
  face_JxW.resize(n_faces, std::vector<double>(n_faces_q_points));
  face_normals.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

  // present_face_velocity_gradients.resize(
  //   n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
  // present_face_pressure_values.resize(n_faces,
  //                                     std::vector<double>(n_faces_q_points));

  exact_solution_full.resize(n_faces_q_points, Vector<double>(n_components));
  grad_exact_solution_full.resize(n_faces_q_points,
                                  std::vector<Tensor<1, dim>>(n_components));
  exact_face_velocity_gradients.resize(
    n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
  exact_face_pressure_values.resize(n_faces,
                                    std::vector<double>(n_faces_q_points));

  phi_u_face.resize(n_faces,
                    std::vector<std::vector<Tensor<1, dim>>>(
                      n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell)));
}

template <int dim>
ScratchDataMonolithicFSI<dim>::ScratchDataMonolithicFSI(
  const FESystem<dim>       &fe,
  const Quadrature<dim>     &cell_quadrature,
  const Mapping<dim>        &fixed_mapping,
  const Mapping<dim>        &moving_mapping,
  const Quadrature<dim - 1> &face_quadrature,
  const unsigned int         dofs_per_cell,
  const unsigned int         boundary_id,
  const std::vector<double> &bdfCoeffs)
  : fe_values(moving_mapping, fe, cell_quadrature, required_updates)
  , fe_values_fixed(fixed_mapping, fe, cell_quadrature, required_updates)
  , fe_face_values(moving_mapping, fe, face_quadrature, required_face_updates)
  , fe_face_values_fixed(fixed_mapping,
                         fe,
                         face_quadrature,
                         required_face_updates)
  , n_q_points(cell_quadrature.size())

  // We assume that simplicial meshes with all tris or tets
  , n_faces((dim == 3) ? 4 : 3)

  , n_faces_q_points(face_quadrature.size())
  , dofs_per_cell(dofs_per_cell)
  , boundary_id(boundary_id)
  , bdfCoeffs(bdfCoeffs)
{
  this->allocate();
}

template <int dim>
ScratchDataMonolithicFSI<dim>::ScratchDataMonolithicFSI(
  const ScratchDataMonolithicFSI &other)
  : fe_values(other.fe_values.get_mapping(),
              other.fe_values.get_fe(),
              other.fe_values.get_quadrature(),
              other.fe_values.get_update_flags())
  , fe_values_fixed(other.fe_values_fixed.get_mapping(),
                    other.fe_values_fixed.get_fe(),
                    other.fe_values_fixed.get_quadrature(),
                    other.fe_values_fixed.get_update_flags())
  , fe_face_values(other.fe_face_values.get_mapping(),
                   other.fe_face_values.get_fe(),
                   other.fe_face_values.get_quadrature(),
                   other.fe_face_values.get_update_flags())
  , fe_face_values_fixed(other.fe_face_values_fixed.get_mapping(),
                         other.fe_face_values_fixed.get_fe(),
                         other.fe_face_values_fixed.get_quadrature(),
                         other.fe_face_values_fixed.get_update_flags())
  , n_q_points(other.n_q_points)
  , n_faces(other.n_faces)
  , n_faces_q_points(other.n_faces_q_points)
  , dofs_per_cell(other.dofs_per_cell)
  , boundary_id(other.boundary_id)
  , bdfCoeffs(other.bdfCoeffs)
{
  this->allocate();
}

template <int dim>
void ScratchDataMonolithicFSI<dim>::allocate()
{
  components.resize(dofs_per_cell);

  JxW_moving.resize(n_q_points);
  JxW_fixed.resize(n_q_points);

  present_velocity_values.resize(n_q_points);
  present_velocity_gradients.resize(n_q_points);
  present_pressure_values.resize(n_q_points);
  present_position_values.resize(n_q_points);
  present_position_gradients.resize(n_q_points);
  present_mesh_velocity_values.resize(n_q_points);

  present_face_velocity_values.resize(
    n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
  present_face_position_values.resize(
    n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
  present_face_position_gradient.resize(
    n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
  present_face_lambda_values.resize(
    n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
  present_face_mesh_velocity_values.resize(
    n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

  source_term_full.resize(n_q_points, Vector<double>(n_components));
  source_term_full_fixed.resize(n_q_points, Vector<double>(n_components));
  source_term_velocity.resize(n_q_points);
  source_term_pressure.resize(n_q_points);
  source_term_position.resize(n_q_points);

  grad_source_term_full.resize(n_q_points,
                               std::vector<Tensor<1, dim>>(n_components));
  grad_source_velocity.resize(n_q_points);
  grad_source_pressure.resize(n_q_points);

  // BDF
  previous_velocity_values.resize(bdfCoeffs.size() - 1,
                                  std::vector<Tensor<1, dim>>(n_q_points));
  previous_position_values.resize(bdfCoeffs.size() - 1,
                                  std::vector<Tensor<1, dim>>(n_q_points));
  previous_face_position_values.resize(
    n_faces,
    std::vector<std::vector<Tensor<1, dim>>>(
      bdfCoeffs.size() - 1, std::vector<Tensor<1, dim>>(n_faces_q_points)));

  phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
  grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
  div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
  phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));
  phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
  grad_phi_x.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
  div_phi_x.resize(n_q_points, std::vector<double>(dofs_per_cell));

  phi_u_face.resize(n_faces,
                    std::vector<std::vector<Tensor<1, dim>>>(
                      n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell)));

  phi_x_face.resize(n_faces,
                    std::vector<std::vector<Tensor<1, dim>>>(
                      n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell)));

  grad_phi_x_face.resize(n_faces,
                         std::vector<std::vector<Tensor<2, dim>>>(
                           n_faces_q_points,
                           std::vector<Tensor<2, dim>>(dofs_per_cell)));

  phi_l_face.resize(n_faces,
                    std::vector<std::vector<Tensor<1, dim>>>(
                      n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell)));

  face_JxW_moving.resize(n_faces, std::vector<double>(n_faces_q_points));
  face_JxW_fixed.resize(n_faces, std::vector<double>(n_faces_q_points));

  face_G.resize(n_faces, std::vector<Tensor<2, dim - 1>>(n_faces_q_points));
  delta_dx.resize(n_faces,
                  std::vector<std::vector<double>>(
                    n_faces_q_points, std::vector<double>(dofs_per_cell)));
}

// Explicit instantiations
template class ScratchDataNS<2>;
template class ScratchDataNS<3>;
template class ScratchDataMonolithicFSI<2>;
template class ScratchDataMonolithicFSI<3>;