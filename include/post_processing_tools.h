#ifndef POST_PROCESSING_TOOLS_H
#define POST_PROCESSING_TOOLS_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <cmath>

/**
 * Collection of utilities for post-processing.
 */

namespace PostProcessingTools
{
  using namespace dealii;

  /**
   * Small specialization of deal.II's DataOutFaces to output a single boundary.
   */

  template <int dim>
  class DataOutFacesOnBoundary : public DataOutFaces<dim>
  {
  public:
    /**
     * Constructor
     */
    DataOutFacesOnBoundary(const Triangulation<dim> &triangulation,
                           const types::boundary_id  boundary_id);

    /**
     * A FaceDescriptor is a pair [cell : number of the face].
     */
    using FaceDescriptor = typename DataOutFaces<dim>::FaceDescriptor;


    FaceDescriptor first_face() override;

    /**
     * Return the next face after which we want output for. If there are no more
     * faces, <tt>dofs->end()</tt> is returned as the first component of the
     * return value.
     */
    FaceDescriptor next_face(const FaceDescriptor &face) override;

  protected:
    const Triangulation<dim> &triangulation;
    const types::boundary_id  boundary_id;
  };

  enum class SliceAxis : unsigned int
  {
    x = 0,
    y = 1,
    z = 2
  };


  /**
   * Compute slice indices for degrees of freedom located on a given boundary.
   *
   *  The boundary @p boundary_id is partitioned into @p n_slices along the
   * coordinate direction @p axis. Each degree of freedom on this boundary
   * is assigned to one slice based on its geometric position.
   *
   * The function returns a Vector<double> containing the slice index
   * associated with each degree of freedom. A floating-point vector is used
   * for compatibility with deal.II data structures and post-processing.
   *
   * @return Vector of slice indices for the specified boundary.
   */
  template <int dim>
  Vector<double>
  compute_slice_index_on_boundary(const DoFHandler<dim>   &dof_handler,
                                  const types::boundary_id boundary_id,
                                  const unsigned int       n_slices,
                                  const SliceAxis          axis,
                                  const MPI_Comm           mpi_comm);

  /**
   * Postprocessor that computes the vorticity from the velocity gradient:
   *   - dim==2: omega = du_y/dx - du_x/dy  (out-of-plane scalar)
   *   - dim==3: omega = ||curl(u)||        (magnitude of the curl, scalar)
   *
   * The postprocessor assumes that the velocity components are stored in the
   * global solution vector starting at component index @p u_first_component.
   *
   * This class is meant for output only (VTU / PVTU / boundary VTU). For source
   * terms during assembly, compute the vorticity directly at quadrature points
   * from FEValues velocity gradients (more direct and cheaper).
   */
  template <int dim>
  class VorticityPostprocessor : public DataPostprocessorScalar<dim>
  {
  public:
    explicit VorticityPostprocessor(const unsigned int u_first_component,
                                    const std::string &name = "vorticity")
      : DataPostprocessorScalar<dim>(name, update_gradients)
      , u_first_component(u_first_component)
    {}

    virtual void
    evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &inputs,
                          std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(computed_quantities.size(), inputs.solution_gradients.size());

      for (unsigned int q = 0; q < computed_quantities.size(); ++q)
      {
        // grad_u[d](j) = d u_d / d x_j
        const auto &grad_u0 = inputs.solution_gradients[q][u_first_component + 0];

        if constexpr (dim == 2)
        {
          const auto &grad_u1 = inputs.solution_gradients[q][u_first_component + 1];
          // omega_z = du_y/dx - du_x/dy
          computed_quantities[q](0) = grad_u1[0] - grad_u0[1];
        }
        else if constexpr (dim == 3)
        {
          const auto &grad_u1 = inputs.solution_gradients[q][u_first_component + 1];
          const auto &grad_u2 = inputs.solution_gradients[q][u_first_component + 2];

          const double omega_x = grad_u2[1] - grad_u1[2];
          const double omega_y = grad_u0[2] - grad_u2[0];
          const double omega_z = grad_u1[0] - grad_u0[1];

          computed_quantities[q](0) =
            std::sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z);
        }
        else
        {
          computed_quantities[q](0) = 0.0;
        }
      }
    }

  private:
    const unsigned int u_first_component;
  };

} // namespace PostProcessingTools

#endif
