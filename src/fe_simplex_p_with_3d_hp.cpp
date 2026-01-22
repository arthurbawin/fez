// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2020 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/polynomials_barycentric.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_orientation.h>

#include <fe_simplex_p_with_3d_hp.h>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_SimplexP_3D_hp<dim, spacedim>::clone() const
{
  return std::make_unique<FE_SimplexP_3D_hp<dim, spacedim>>(*this);
}

template <int dim, int spacedim>
std::string
FE_SimplexP_3D_hp<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_SimplexP_3D_hp<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree << ")";

  return namebuf.str();
}

template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_SimplexP_3D_hp<dim, spacedim>::hp_vertex_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  // AssertDimension(dim, 2);

  if (dynamic_cast<const FE_SimplexP_3D_hp<dim, spacedim> *>(&fe_other) != nullptr)
    {
      // there should be exactly one single DoF of each FE at a vertex, and
      // they should have identical value
      return {{0U, 0U}};
    }
  else if (dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other) != nullptr)
    {
      // there should be exactly one single DoF of each FE at a vertex, and
      // they should have identical value
      return {{0U, 0U}};
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
    {
      // the FE_Nothing has no degrees of freedom, so there are no
      // equivalencies to be recorded
      return {};
    }
  else if (fe_other.n_unique_faces() == 1 && fe_other.n_dofs_per_face(0) == 0)
    {
      // if the other element has no DoFs on faces at all,
      // then it would be impossible to enforce any kind of
      // continuity even if we knew exactly what kind of element
      // we have -- simply because the other element declares
      // that it is discontinuous because it has no DoFs on
      // its faces. in that case, just state that we have no
      // constraints to declare
      return {};
    }
  else
    {
      DEAL_II_NOT_IMPLEMENTED();
      return {};
    }
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_SimplexP_3D_hp<dim, spacedim>::hp_line_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  // AssertDimension(dim, 2);

  if (const FE_SimplexP_3D_hp<dim, spacedim> *fe_p_other =
        dynamic_cast<const FE_SimplexP_3D_hp<dim, spacedim> *>(&fe_other))
    {
      // dofs are located along lines, so two dofs are identical if they are
      // located at identical positions.
      // Therefore, read the points in unit_support_points for the
      // first coordinate direction. For FE_SimplexP, they are currently
      // hard-coded and we iterate over points on the first line which begin
      // after the 3 vertex points in the complete list of unit support points

      std::vector<std::pair<unsigned int, unsigned int>> identities;

      for (unsigned int i = 0; i < this->degree - 1; ++i)
        for (unsigned int j = 0; j < fe_p_other->degree - 1; ++j)
          if (std::fabs(this->unit_support_points[i + 3][0] -
                        fe_p_other->unit_support_points[i + 3][0]) < 1e-14)
            identities.emplace_back(i, j);
          else
            {
              // If nodes are not located in the same place, we have to
              // interpolate. This is then not handled through the
              // current function, but via interpolation matrices that
              // result in constraints, rather than identities. Since
              // that happens in a different function, there is nothing
              // for us to do here.
            }

      return identities;
    }
  else if (const FE_Q<dim, spacedim> *fe_q_other =
             dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other))
    {
      // dofs are located along lines, so two dofs are identical if they are
      // located at identical positions. if we had only equidistant points, we
      // could simply check for similarity like (i+1)*q == (j+1)*p, but we
      // might have other support points (e.g. Gauss-Lobatto
      // points). Therefore, read the points in unit_support_points for the
      // first coordinate direction. For FE_Q, we take the lexicographic
      // ordering of the line support points in the first direction (i.e.,
      // x-direction), which we access between index 1 and p-1 (index 0 and p
      // are vertex dofs). For FE_SimplexP, they are currently hard-coded and we
      // iterate over points on the first line which begin after the 3 vertex
      // points in the complete list of unit support points

      const std::vector<unsigned int> &index_map_inverse_q_other =
        fe_q_other->get_poly_space_numbering_inverse();

      std::vector<std::pair<unsigned int, unsigned int>> identities;

      for (unsigned int i = 0; i < this->degree - 1; ++i)
        for (unsigned int j = 0; j < fe_q_other->degree - 1; ++j)
          if (std::fabs(this->unit_support_points[i + 3][0] -
                        fe_q_other->get_unit_support_points()
                          [index_map_inverse_q_other[j + 1]][0]) < 1e-14)
            identities.emplace_back(i, j);
          else
            {
              // If nodes are not located in the same place, we have to
              // interpolate. This will then also
              // capture the case where the FE_Q has a different polynomial
              // degree than the current element. In either case, the resulting
              // constraints are computed elsewhere, rather than via the
              // identities this function returns: Since
              // that happens in a different function, there is nothing
              // for us to do here.
            }

      return identities;
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
    {
      // The FE_Nothing has no degrees of freedom, so there are no
      // equivalencies to be recorded. (If the FE_Nothing is dominating,
      // then this will also leads to constraints, but we are not concerned
      // with this here.)
      return {};
    }
  else if (fe_other.n_unique_faces() == 1 && fe_other.n_dofs_per_face(0) == 0)
    {
      // if the other element has no elements on faces at all,
      // then it would be impossible to enforce any kind of
      // continuity even if we knew exactly what kind of element
      // we have -- simply because the other element declares
      // that it is discontinuous because it has no DoFs on
      // its faces. in that case, just state that we have no
      // constraints to declare
      return {};
    }
  else
    {
      DEAL_II_NOT_IMPLEMENTED();
      return {};
    }
}

template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_SimplexP_3D_hp<dim, spacedim>::hp_quad_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other,
  const unsigned int) const
{
  return std::vector<std::pair<unsigned int, unsigned int>>();
  
  // // we can presently only compute these identities if both FEs are FE_Qs or
  // // if the other one is an FE_Nothing
  // if (const FE_SimplexP_3D_hp<dim, spacedim> *fe_q_other =
  //       dynamic_cast<const FE_SimplexP_3D_hp<dim, spacedim> *>(&fe_other))
  //   {
  //     // this works exactly like the line case above, except that now we have
  //     // to have two indices i1, i2 and j1, j2 to characterize the dofs on the
  //     // face of each of the finite elements. since they are ordered
  //     // lexicographically along the first line and we have a tensor product,
  //     // the rest is rather straightforward
  //     const unsigned int p = this->degree;
  //     const unsigned int q = fe_q_other->degree;

  //     std::vector<std::pair<unsigned int, unsigned int>> identities;

  //     const std::vector<unsigned int> &index_map_inverse =
  //       this->get_poly_space_numbering_inverse();
  //     const std::vector<unsigned int> &index_map_inverse_other =
  //       fe_q_other->get_poly_space_numbering_inverse();

  //     for (unsigned int i1 = 0; i1 < p - 1; ++i1)
  //       for (unsigned int i2 = 0; i2 < p - 1; ++i2)
  //         for (unsigned int j1 = 0; j1 < q - 1; ++j1)
  //           for (unsigned int j2 = 0; j2 < q - 1; ++j2)
  //             if ((std::fabs(
  //                    this->unit_support_points[index_map_inverse[i1 + 1]][0] -
  //                    fe_q_other
  //                      ->unit_support_points[index_map_inverse_other[j1 + 1]]
  //                                           [0]) < 1e-14) &&
  //                 (std::fabs(
  //                    this->unit_support_points[index_map_inverse[i2 + 1]][0] -
  //                    fe_q_other
  //                      ->unit_support_points[index_map_inverse_other[j2 + 1]]
  //                                           [0]) < 1e-14))
  //               identities.emplace_back(i1 * (p - 1) + i2, j1 * (q - 1) + j2);

  //     return identities;
  //   }
  // else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
  //   {
  //     // the FE_Nothing has no degrees of freedom, so there are no
  //     // equivalencies to be recorded
  //     return std::vector<std::pair<unsigned int, unsigned int>>();
  //   }
  // else if (fe_other.n_unique_faces() == 1 && fe_other.n_dofs_per_face(0) == 0)
  //   {
  //     // if the other element has no elements on faces at all,
  //     // then it would be impossible to enforce any kind of
  //     // continuity even if we knew exactly what kind of element
  //     // we have -- simply because the other element declares
  //     // that it is discontinuous because it has no DoFs on
  //     // its faces. in that case, just state that we have no
  //     // constraints to declare
  //     return std::vector<std::pair<unsigned int, unsigned int>>();
  //   }
  // else
  //   {
  //     DEAL_II_NOT_IMPLEMENTED();
  //     return std::vector<std::pair<unsigned int, unsigned int>>();
  //   }
}

// explicit instantiations
template class FE_SimplexP_3D_hp<2, 2>;
template class FE_SimplexP_3D_hp<2, 3>;
template class FE_SimplexP_3D_hp<3, 3>;

DEAL_II_NAMESPACE_CLOSE
