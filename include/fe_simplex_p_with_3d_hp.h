// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2021 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_fe_fe_p_with_3d_hp_h
#define dealii_fe_fe_p_with_3d_hp_h

#include <deal.II/fe/fe_simplex_p.h>

DEAL_II_NAMESPACE_OPEN

/**
 *
 */
template <int dim, int spacedim = dim>
class FE_SimplexP_3D_hp : public FE_SimplexP<dim, spacedim>
{
public:
  /**
   * Constructor.
   */
  FE_SimplexP_3D_hp(const unsigned int degree)
    : FE_SimplexP<dim, spacedim>(degree)
  {}

  /**
   * @copydoc dealii::FiniteElement::clone()
   */
  std::unique_ptr<FiniteElement<dim, spacedim>> clone() const override;

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_SimplexP<dim>(degree)</tt>, with @p dim and @p degree
   * replaced by appropriate values.
   */
  std::string get_name() const override;

  /**
   * @copydoc dealii::FiniteElement::hp_vertex_dof_identities()
   */
  std::vector<std::pair<unsigned int, unsigned int>> hp_vertex_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  /**
   * @copydoc dealii::FiniteElement::hp_line_dof_identities()
   */
  std::vector<std::pair<unsigned int, unsigned int>> hp_line_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  std::vector<std::pair<unsigned int, unsigned int>>
  hp_quad_dof_identities(const FiniteElement<dim, spacedim> &fe_other,
                         const unsigned int) const override;
};

DEAL_II_NAMESPACE_CLOSE

#endif
