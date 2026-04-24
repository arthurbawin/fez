#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <parsed_function_symengine.h>
#include <solver_info.h>

#define DECLARE_VERBOSITY_PARAM(prm, default_verbosity)                        \
  (prm).declare_entry("verbosity",                                             \
                      std::string(default_verbosity),                          \
                      Patterns::Selection("quiet|verbose"),                    \
                      "Level of message display in console: quiet or verbose " \
                      "(default: " +                                           \
                        std::string(default_verbosity) + ")");

#define READ_VERBOSITY_PARAM(prm, verbosity)                     \
  {                                                              \
    const std::string parsed_verbosity = (prm).get("verbosity"); \
    if (parsed_verbosity == "quiet")                             \
      verbosity = Verbosity::quiet;                              \
    if (parsed_verbosity == "verbose")                           \
      verbosity = Verbosity::verbose;                            \
  }

/**
 * This namespace contains the parameters used to control the various
 * parts of the solvers : mesh, (non-)linear solver, time integration, etc.
 */
namespace Parameters
{
  using namespace dealii;

  /**
   * Verbosity is set to "verbose" by default for all structures.
   */
  enum class Verbosity
  {
    quiet,
    verbose
  };

  // The problem dimension is read in a first pass to instantiate the right
  // pre-compiled solver. Because dimension in deal.II is not a simulation
  // parameters per se, this is not done here, but in utilities.h. However, the
  // "Dimension" block read by the function in utilities.h should still be read
  // in the real run to avoid an exception. This is done here, and the dimension
  // is not parsed.
  struct DummyDimension
  {
    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Timer
  {
    bool enable_timer;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct BoundaryConditionsData
  {
    Verbosity fluid_verbosity;

    // These are parsed in utilities.h
    unsigned int n_fluid_bc         = 0;
    unsigned int n_pseudosolid_bc   = 0;
    unsigned int n_cahn_hilliard_bc = 0;
    unsigned int n_heat_bc          = 0;

    // FIXME: This is not BC related, maybe move this in a dedicated entity
    unsigned int n_metric_fields = 0;

    bool fix_pressure_constant;
    bool enforce_zero_mean_pressure;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Mesh
  {
    Verbosity verbosity;

    // Gmsh mesh file
    std::string filename;

    bool         use_deal_ii_cube_mesh;
    std::string  deal_ii_preset_mesh;
    std::string  deal_ii_mesh_param;
    unsigned int refinement_level;

    // Name of each mesh physical entities
    std::map<types::boundary_id, std::string> id2name;
    std::map<std::string, types::boundary_id> name2id;

    /**
     * Parameters controlling the mesh adaptation procedure
     */
    struct Adaptation
    {
      Verbosity verbosity;

      bool enable;

      // Directory into which mesh adaptation-related files are written
      std::string adapt_dir;

      // Extension for the adapted meshes
      std::string adapted_mesh_extension;

      /**
       * Available mesh adaptation strategies:
       * - adaptation with a Riemannian metric, originating from one or more FE
       * fields. For simplicial meshes only, as anisotropic metric-based meshing
       * libraries exist only for simplicial meshes for now.
       * - (not yet implemented:) hierarchical adaptation, using deal.II's
       * routines and p4est. For quad/hex meshes only.
       */
      enum class Strategy
      {
        RiemannianMetric
      } strategy;

      /**
       * Parameters for mesh adaptation with a Riemannian metric
       */
      struct Metric
      {
        /**
         * Number of fixed point iterations to perform, to converge the
         * mesh-solution pair. For steady simulations, this is the number of
         * times the solver is run, and the mesh is adapted this number of
         * times minus one. For unsteady simulations, this is the number of
         * times the whole simulation is run, and the meshes are adapted on
         * sub-intervals, this number of times minus one.
         */
        unsigned int n_fixed_point;

        unsigned int current_fixed_point_iteration;

        // Level of verbosity of the MMG library
        unsigned int mmg_verbosity;

        // For steady simulations, specify whether the solution should be
        // transferred (projected) from the initial mesh to the adapted mesh.
        bool transfer_solution;
      } metric;

    } adaptation;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Output
  {
    bool         write_results;
    std::string  output_dir;
    std::string  output_prefix;
    unsigned int vtu_output_frequency;

    // A "skin" is a codimension 1 boundary on which we wish to extract data
    // for visualization and/or postprocessing
    struct Skin
    {
      bool               write_results;
      types::boundary_id boundary_id;
      std::string        output_prefix;
      unsigned int       output_frequency;
    } skin;

    static void declare_parameters(ParameterHandler &prm);
    void        read_parameters(ParameterHandler &prm);
  };

  struct PostProcessing
  {
    // A small base struct for postprocessed quantities which can be
    // outputted to a file
    struct PostProcessingBase
    {
      Verbosity verbosity;

      // Enable/disable this postprocessing
      bool enable;

      // Output the results of this postprocessing to a file
      bool         write_results;
      std::string  output_prefix;
      unsigned int output_frequency;
      unsigned int precision;
    };

    // Derived class for postprocessing on a boundary
    struct PostProcessingBaseBoundary : public PostProcessingBase
    {
      types::boundary_id boundary_id;
    };

    // Hydrodynamic forces on a single boundary
    struct Forces : public PostProcessingBaseBoundary
    {
      // The method used to evaluate the forces on a boundary
      enum class ComputationMethod
      {
        stress_vector,
        lagrange_multiplier
      } method;
    } forces;

    // For the FSI solver, compute and export the position of the structure's
    // geometric center.
    struct StructurePosition : public PostProcessingBaseBoundary
    {
      // No additional members for now
    } structure_position;

    // Cut structure into slices and compute forces on each individual slice
    // Used e.g. to measure correlation of forces coefficients along cylinder
    struct Slices : public PostProcessingBaseBoundary
    {
      std::string  along_which_axis;
      unsigned int n_slices;
      bool         compute_forces_on_slices;
    } slices;

    static void declare_parameters(ParameterHandler &prm);
    void        read_parameters(ParameterHandler &prm);
  };

  template <int dim>
  struct FiniteElements
  {
    // If true, use hypercubes, otherwise use simplices (default).
    bool use_quads;

    // If true, enable residual-based stabilization terms.
    bool stabilization;

    // Degree of the velocity interpolation
    unsigned int velocity_degree;

    // Degree of the pressure interpolation
    unsigned int pressure_degree;

    // Degree of the mesh position interpolation
    unsigned int mesh_position_degree;

    // Degree of the Lagrange multipliers interpolation
    // when enforcing weak no-slip constraints
    unsigned int no_slip_lagrange_mult_degree;

    // Degree of the tracer and potential interpolation for two-phase
    // flows with a Cahn-Hilliard Navier-Stokes model
    unsigned int tracer_degree;
    unsigned int potential_degree;

    // Degree of the temperature for the heat equation
    unsigned int temperature_degree;

    struct QuadratureRule
    {
      enum class Type
      {
        GaussSimplex,
        WitherdenVincent
      } type;

      unsigned int n_pts_1D_simplex_cell_quad;
      unsigned int n_pts_1D_simplex_face_quad;
    };

    QuadratureRule rule;
    QuadratureRule rule_for_error;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Fluid
  {
    double density;
    double kinematic_viscosity;
    double dynamic_viscosity;

    void declare_parameters(ParameterHandler &prm, unsigned int index);
    void read_parameters(ParameterHandler &prm, unsigned int index);
  };

  template <int dim>
  class PseudoSolid
  {
  public:
    enum class ConstitutiveModel
    {
      linear_elasticity,
      neo_hookean,
      HN_0,
      HN_1,
      Ogden_1,
      Ogden_2,
      Ogden_2_classique,
      quad,
    };

    ConstitutiveModel constitutive_model = ConstitutiveModel::linear_elasticity;

    std::shared_ptr<ManufacturedSolutions::ParsedFunctionSDBase<dim>>
      lame_lambda_fun;
    std::shared_ptr<ManufacturedSolutions::ParsedFunctionSDBase<dim>>
      lame_mu_fun;

  public:
    void set_time(const double newtime)
    {
      lame_lambda_fun->set_time(newtime);
      lame_mu_fun->set_time(newtime);
    }
    void declare_parameters(ParameterHandler &prm, unsigned int index);
    void read_parameters(ParameterHandler &prm, unsigned int index);
  };

  template <int dim>
  class PhysicalProperties
  {
  public:
    const unsigned int max_fluids = 2;
    unsigned int       n_fluids;
    std::vector<Fluid> fluids;

    const unsigned int            max_pseudosolids = 1;
    unsigned int                  n_pseudosolids;
    std::vector<PseudoSolid<dim>> pseudosolids;

  public:
    void set_time(const double newtime)
    {
      for (auto &ps : pseudosolids)
        ps.set_time(newtime);
    }
    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct NonLinearSolver
  {
    double       tolerance;
    double       divergence_tolerance;
    unsigned int max_iterations;
    bool         enable_line_search;
    bool         analytic_jacobian;
    Verbosity    verbosity;

    double reassembly_decrease_tol;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct LinearSolver
  {
    Verbosity verbosity;

    enum class Method
    {
      direct_mumps,
      cg,
      gmres
    } method;

    // Tolerance and max number of iterations for iterative solvers
    double       tolerance;
    unsigned int max_iterations;

    // Fill-in levels for ILU preconditioner
    unsigned int ilu_fill_level;

    /**
     * When using MUMPS as solver, "reuse" the symbolic factorization of the
     * system matrix across the solves. If the sparsity pattern does not change,
     * then the symbolic factorization can be conserved, saving time.
     *
     * This is done through an extension of deal.II's PETSc interface to MUMPS,
     * which, as a beneficial side effect, also checks the MUMPS error code,
     * which is not done in deal.II. This allows throwing an error when the
     * matrix is singular, instead of getting "nan" results.
     *
     * TODO: the associated MUMPS solver should be looked into, as it is
     * unclear that the factorization is indeed reused and/or that it is more
     * efficient. Unlike Pardiso, the symbolic factorization step is not cleanly
     * separated from the actual factorization and solve steps.
     */
    bool reuse;

    void declare_parameters(ParameterHandler  &prm,
                            const std::string &solver_type);
    void read_parameters(ParameterHandler &prm, const std::string &solver_type);
  };

  struct TimeIntegration
  {
    Verbosity verbosity;

    double dt;
    double t_initial;
    double t_end;

    enum class Scheme
    {
      stationary,
      BDF1,
      BDF2
    } scheme;

    enum class BDFStart
    {
      BDF1,
      initial_condition
    } bdfstart;

    // For BDF2 scheme using BDF1 a starting scheme, the BDF1 step is
    // done with this value times the initial time step.
    double bdf_starting_step_ratio;

    struct Adaptation
    {
      Verbosity verbosity;

      bool enable;

      // Implemented strategies for time step adaptation:
      // - adapt based on an estimate of the BDF truncation error
      // - adapt based on the maximum CFL number (only for solvers with
      //   a velocity variable)
      enum class AdaptationStrategy
      {
        BDFTruncationError,
        CFL
      } strategy;

      double max_timestep;
      double min_timestep;
      double max_timestep_increase;
      double max_timestep_reduction;

      // Parameters for adaptation based on BDF truncation error
      std::map<SolverInfo::VariableType, double> target_error;
      bool   reject_timestep_with_large_error;
      double reject_error_factor;

      // Parameters for adaptation based on CFL
      double target_cfl;
      bool   reject_timestep_with_large_cfl;
      double reject_cfl_factor;

      // FIXME: Both parameters below are currently unused:
      // required_times because it is tricky to adjust or merge the time steps
      // to reach the required times without considering corner cases, and
      // compute_error_on_estimator because we may or may not want to compute
      // the convergence of the error estimator w.r.t. the true error.

      // Required times : the simulation must absolutely go through these
      std::vector<double> required_times;
      bool                compute_error_on_estimator;
    } adaptation;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
    bool is_steady() const { return scheme == Scheme::stationary; }
  };

  template <int dim>
  class CahnHilliard
  {
  public:
    enum class MobilityModel
    {
      constant
    } mobility_model;

    enum class MeshForcingLaw
    {
      simple,
      regularized_band
    } mesh_forcing_law = MeshForcingLaw::regularized_band;

    double mobility;
    double surface_tension;
    double epsilon_interface;
    double epsilon_interface_enlarged;
    double psi_interface_width_factor;
    bool   with_tracer_limiter;

    // Mesh forcing parameters : these parameters control the behavior of the
    // source term in the pseudosolid equation, in the CHNS-ALE model.
    double mff_enlarged_compression_factor;
    double mff_physics_compression_factor;
    double mff_transport_factor;
    double mff_band_factor;
    /**
     * We differentiate between the body force which is multiplied by the
     * mixture density (typically gravity), and the generic source term (e.g.,
     * for manufactured solutions) which is not.
     */
    Tensor<1, dim> body_force;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct LinearElasticity
  {
    // If true, then the provided position source term is to be evaluated on
    // the current mesh, and not on the reference mesh where the elasticity
    // equation is solved (that is, we evaluate f(x(X)) instead of f(X).
    bool enable_source_term_on_current_mesh;

    // The source term on current mesh is enforced with a continuation method,
    // starting at min_coeff * f(x(X)) and progressing until max_coeff * f(x(X))
    double min_current_mesh_source_term_multiplier;
    double max_current_mesh_source_term_multiplier;

    // Number of steps to use in the continuation method when the source term
    // is applied on the current configuration.
    unsigned int n_continuation_steps;

    // If true, runs the linear elasticity solver as a pre-processing step
    // to compute an initial mesh deformation. The resulting position field
    // is used to initialize the ALE mesh of the CHNS solver, typically when
    // mesh forcing is activated.
    bool use_as_presolver;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct CheckpointRestart
  {
    bool enable_checkpoint;
    // If true, restart simulation from the given checkpoint file
    bool restart;
    // Name of the file to write to/read when checkpointing/restarting resp.
    std::string filename;
    // Write checkpoint every N time steps
    unsigned int checkpoint_frequency;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct MMS
  {
    bool enable;

    // The type of study to perform : refinement in space and/or time
    enum class Type
    {
      space,
      time,
      spacetime
    } type;

    // The Lp norm used to compute the error in time
    // The norm for the error in space is given through norms_to_compute
    // FIXME: do the same for the time
    enum class TimeLpNorm
    {
      L1,
      L2,
      Linfty
    } time_norm;

    // Subtract the mean value from the exact pressure solution.
    // This must be enabled when performing a convergence study while also
    // enforcing a zero-mean pressure solution, otherwise both functions
    // differ by the constant mean.
    bool subtract_mean_pressure;

    // Force the use of the provided source term in the "Source terms" section,
    // even during a convergence study. This can be used when the provided exact
    // solution is really a solution of the system of PDEs for the given source
    // terms. In that case, the source terms obtained from the MMS are not set.
    bool force_source_term;

    unsigned int n_convergence;
    unsigned int current_step = 0;
    int          run_only_step;

    // FIXME: remove these, and use only the options from the "Mesh" section
    bool use_deal_ii_cube_mesh;
    bool use_deal_ii_holed_plate_mesh;

    std::string  mesh_prefix;
    unsigned int first_mesh_index;
    unsigned int mesh_suffix = 0;

    std::vector<VectorTools::NormType> norms_to_compute;

    bool         use_space_convergence_mesh;
    unsigned int spatial_mesh_index;
    double       time_step_reduction_factor;

    // Options to write the convergence rates to a file
    bool        write_convergence_table_to_file;
    std::string convergence_file_prefix;
    bool        compute_rates_only_at_end;

    // Print the errors for each time step in console
    bool        print_unsteady_errors_to_console;
    bool        print_unsteady_errors_to_file;
    std::string unsteady_errors_file_prefix;

    // For anisotropic mesh adaptation, the target number of vertices for the
    // current convergence step
    // FIXME: the GenericSolver should use the full parameters and modify the
    // metric field parameters instead of duplicating this information
    unsigned int n_target_vertices;

    void override_mesh_filename(Mesh &mesh_param, const unsigned int index)
    {
      mesh_param.filename = mesh_prefix + std::to_string(index) + ".msh";
    }

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  /**
   * Fluid-structure interaction
   */
  template <int dim>
  struct FSI
  {
    Verbosity verbosity;

    bool   enable_coupling;
    double spring_constant;
    double damping;
    double mass;

    double cylinder_radius;
    double cylinder_length;

    Point<dim> cylinder_center;

    bool fix_z_component;

    bool compute_error_on_forces;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  /**
   * Options for debugging
   */
  struct Debug
  {
    Verbosity    verbosity;
    bool         write_dealii_mesh_as_msh;
    bool         write_partition_pos_gmsh;
    bool         apply_exact_solution;
    bool         compare_analytical_jacobian_with_fd;
    double       analytical_jacobian_absolute_tolerance;
    double       analytical_jacobian_relative_tolerance;
    bool         fsi_apply_erroneous_coupling;
    bool         fsi_check_mms_on_boundary;
    unsigned int fsi_coupling_option;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };
} // namespace Parameters
#include "pseudosolid_material.impl.h"
#endif
