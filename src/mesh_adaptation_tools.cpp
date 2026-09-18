
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_out.h>
#include <mesh.h>
#include <mesh_adaptation_tools.h>
#include <metric_field.h>
#include <parameter_reader.h>

#include <algorithm>
#include <array>
#include <limits>
#include <set>

#if defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#endif

#if defined(FEZ_WITH_MMG)
#  include <mmg/libmmg.h>
#endif

namespace MeshTools
{
  template <int dim>
  void
  mark_multifield_adaptation(parallel::DistributedTriangulationBase<dim> &tria,
                             const std::vector<Vector<float>> &indicators,
                             const Parameters::Mesh::Adaptation::TreeAMR &param)
  {
    const auto                comm = tria.get_mpi_communicator();
    std::vector<unsigned int> votes(tria.n_active_cells(), 0);
    std::vector<bool>         coarsen(tria.n_active_cells(), true);
    unsigned int              active_fields = 0;
    for (const auto &indicator : indicators)
    {
      AssertDimension(indicator.size(), tria.n_active_cells());
      const double maximum = Utilities::MPI::max(
        indicator.size() == 0 ? 0.f : indicator.linfty_norm(), comm);
      if (maximum == 0.)
        continue;
      ++active_fields;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          cell->clear_refine_flag();
          cell->clear_coarsen_flag();
        }
      using Strategy =
        Parameters::Mesh::Adaptation::TreeAMR::RefinementStrategy;
      if (param.refinement_strategy == Strategy::FixedNumber)
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
          tria, indicator, param.fraction_to_refine, param.fraction_to_coarsen);
      else
      {
        AssertThrow(param.refinement_strategy == Strategy::FixedFraction,
                    ExcMessage(
                      "Independent field selection requires a Kelly strategy"));
        parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction(tria,
                                            indicator,
                                            param.fraction_to_refine,
                                            param.fraction_to_coarsen);
      }
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          const auto i = cell->active_cell_index();
          votes[i] +=
            cell->refine_flag_set() != RefinementCase<dim>::no_refinement;
          coarsen[i] = coarsen[i] && cell->coarsen_flag_set();
        }
    }

    // Apply the budget once, after combining decisions. Do not turn a field's
    // veto into forced coarsening to satisfy the budget. Balancing may still
    // add cells, as in the single-field deal.II path.
    std::vector<std::pair<unsigned int, std::string>> requests;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        cell->clear_refine_flag();
        cell->clear_coarsen_flag();
        const auto i = cell->active_cell_index();
        if (votes[i] > 0 &&
            static_cast<unsigned int>(cell->level()) < param.max_level)
          requests.emplace_back(active_fields - votes[i],
                                cell->id().to_string());
        if (active_fields > 0 && votes[i] == 0 && coarsen[i] &&
            static_cast<unsigned int>(cell->level()) > param.min_level)
          cell->set_coarsen_flag();
      }
    const auto count   = tria.n_global_active_cells();
    const auto budget  = param.max_n_cells > count ?
                           (param.max_n_cells - count) /
                            (GeometryInfo<dim>::max_children_per_cell - 1) :
                           0;
    const bool limited = Utilities::MPI::sum(requests.size(), comm) > budget;
    std::set<std::string> selected;
    if (limited)
    {
      std::vector<std::pair<unsigned int, std::string>> global_requests;
      for (const auto &rank : Utilities::MPI::all_gather(comm, requests))
        global_requests.insert(global_requests.end(), rank.begin(), rank.end());
      // More fields requesting refinement take priority; CellId breaks ties
      // independently of the MPI partition and without comparing field units.
      std::sort(global_requests.begin(), global_requests.end());
      global_requests.resize(budget);
      for (const auto &request : global_requests)
        selected.insert(request.second);
      if (Utilities::MPI::this_mpi_process(comm) == 0)
        std::cout << "Multifield AMR: cell budget limits the union of "
                     "refinement requests."
                  << std::endl;
    }
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && votes[cell->active_cell_index()] > 0 &&
          static_cast<unsigned int>(cell->level()) < param.max_level &&
          (!limited || selected.count(cell->id().to_string())))
        cell->set_refine_flag();
  }

  template void
  mark_multifield_adaptation(parallel::DistributedTriangulationBase<2> &,
                             const std::vector<Vector<float>> &,
                             const Parameters::Mesh::Adaptation::TreeAMR &);
  template void
  mark_multifield_adaptation(parallel::DistributedTriangulationBase<3> &,
                             const std::vector<Vector<float>> &,
                             const Parameters::Mesh::Adaptation::TreeAMR &);

  namespace
  {
    using Segment  = std::array<Point<2>, 2>;
    using Triangle = std::array<Point<2>, 3>;

    double cross(const Tensor<1, 2> &a, const Tensor<1, 2> &b)
    {
      return a[0] * b[1] - a[1] * b[0];
    }

    double point_segment_distance(const Point<2> &p, const Segment &s)
    {
      const auto   direction = s[1] - s[0];
      const double length    = direction.norm_square();
      const double t =
        length > 0. ? std::clamp((p - s[0]) * direction / length, 0., 1.) : 0.;
      return p.distance(s[0] + t * direction);
    }

    double segment_distance(const Segment &a, const Segment &b)
    {
      const auto   u = a[1] - a[0], v = b[1] - b[0];
      const double denominator = cross(u, v);
      if (denominator != 0.)
      {
        const double t = cross(b[0] - a[0], v) / denominator;
        const double s = cross(b[0] - a[0], u) / denominator;
        if (t >= 0. && t <= 1. && s >= 0. && s <= 1.)
          return 0.;
      }
      return std::min({point_segment_distance(a[0], b),
                       point_segment_distance(a[1], b),
                       point_segment_distance(b[0], a),
                       point_segment_distance(b[1], a)});
    }

    double triangle_distance(const Triangle &t, const Segment &s)
    {
      // A segment can be entirely inside the triangle without cutting an edge.
      if (cross(t[1] - t[0], t[2] - t[0]) != 0.)
        for (const auto &p : s)
        {
          const double a = cross(t[1] - t[0], p - t[0]);
          const double b = cross(t[2] - t[1], p - t[1]);
          const double c = cross(t[0] - t[2], p - t[2]);
          if ((a >= 0. && b >= 0. && c >= 0.) ||
              (a <= 0. && b <= 0. && c <= 0.))
            return 0.;
        }
      return std::min({segment_distance({{t[0], t[1]}}, s),
                       segment_distance({{t[1], t[2]}}, s),
                       segment_distance({{t[2], t[0]}}, s)});
    }
  } // namespace

  template <bool with_moving_mesh>
  void compute_interface_band_criterion(const DoFHandler<2>     &dof_handler,
                                        const Mapping<2>        &mapping,
                                        const LA::ParVectorType &solution,
                                        const unsigned int       component,
                                        const double             half_width,
                                        const double   target_diameter,
                                        Vector<float> &criteria)
  {
    AssertThrow(std::isfinite(half_width) && half_width > 0. &&
                  std::isfinite(target_diameter) && target_diameter > 0.,
                ExcMessage("interface band requires positive finite physical "
                           "width and target diameter"));
    const auto &fe = dof_handler.get_fe();
    AssertThrow(fe.reference_cell() == ReferenceCells::Quadrilateral,
                ExcMessage("interface band requires quadrilateral cells"));
    const unsigned int               subdivisions = std::max(2u, fe.degree);
    const QIterated<2>               samples(QTrapezoid<1>(), subdivisions);
    FEValues<2>                      values(mapping,
                       fe,
                       samples,
                       update_values | update_quadrature_points);
    const FEValuesExtractors::Scalar tracer(component);
    std::vector<double>              phi(samples.size());
    std::vector<std::array<unsigned int, 3>> triangles;
    for (unsigned int j = 0; j < subdivisions; ++j)
      for (unsigned int i = 0; i < subdivisions; ++i)
      {
        const unsigned int a = j * (subdivisions + 1) + i;
        const unsigned int b = a + 1, c = a + subdivisions + 1, d = c + 1;
        triangles.push_back({{a, b, d}});
        triangles.push_back({{a, d, c}});
      }

    // Gather only locally owned contour pieces, including interfaces lying on
    // cell edges. Flat coordinates keep MPI serialization straightforward.
    std::vector<double> local_segments;
    const auto          append = [&local_segments](const Point<2> &a,
                                          const Point<2> &b) {
      local_segments.insert(local_segments.end(), {a[0], a[1], b[0], b[1]});
    };
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        values[tracer].get_function_values(solution, phi);
        const auto &points = values.get_quadrature_points();
        for (const auto &indices : triangles)
        {
          std::vector<Point<2>> cuts;
          for (unsigned int k = 0; k < 3; ++k)
          {
            const auto a = indices[k], b = indices[(k + 1) % 3];
            if (phi[a] == 0.)
              cuts.push_back(points[a]);
            if ((phi[a] < 0. && phi[b] > 0.) || (phi[a] > 0. && phi[b] < 0.))
              cuts.push_back(points[a] + phi[a] / (phi[a] - phi[b]) *
                                           (points[b] - points[a]));
          }
          // Three zero vertices mean the whole triangle is in the zero set.
          if (cuts.size() == 1)
            append(cuts[0], cuts[0]);
          else if (cuts.size() >= 2)
          {
            append(cuts[0], cuts[1]);
            if (cuts.size() == 3)
            {
              append(cuts[1], cuts[2]);
              append(cuts[2], cuts[0]);
            }
          }
        }
      }
    const auto gathered =
      Utilities::MPI::all_gather(dof_handler.get_mpi_communicator(),
                                 local_segments);
    std::vector<Segment> segments;
    for (const auto &coordinates : gathered)
      for (unsigned int i = 0; i < coordinates.size(); i += 4)
        segments.push_back(
          {{Point<2>(coordinates[i], coordinates[i + 1]),
            Point<2>(coordinates[i + 2], coordinates[i + 3])}});

    criteria.reinit(dof_handler.get_triangulation().n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        const auto &points   = values.get_quadrature_points();
        double      distance = std::numeric_limits<double>::infinity();
        Point<2>    lower = points.front(), upper = points.front();
        for (const auto &point : points)
          for (unsigned int d = 0; d < 2; ++d)
          {
            lower[d] = std::min(lower[d], point[d]);
            upper[d] = std::max(upper[d], point[d]);
          }
        for (const auto &segment : segments)
        {
          double box_distance_squared = 0.;
          for (unsigned int d = 0; d < 2; ++d)
          {
            const double gap =
              std::max({0.,
                        lower[d] - std::max(segment[0][d], segment[1][d]),
                        std::min(segment[0][d], segment[1][d]) - upper[d]});
            box_distance_squared += gap * gap;
          }
          if (box_distance_squared > 1.21 * half_width * half_width)
            continue;
          for (const auto &indices : triangles)
          {
            const Triangle triangle{
              {points[indices[0]], points[indices[1]], points[indices[2]]}};
            distance = std::min(distance, triangle_distance(triangle, segment));
          }
          if (distance <= half_width)
            break;
        }
        auto  &criterion = criteria[cell->active_cell_index()];
        double diameter  = cell->diameter();
        if constexpr (with_moving_mesh)
          diameter = mapped_cell_diameter(mapping, cell);
        if (distance <= half_width && diameter > target_diameter * (1. + 1e-12))
          criterion = 1.f;
        else if (cell->level() > 0)
        {
          double parent_diameter = cell->parent()->diameter();
          if constexpr (with_moving_mesh)
            // An inactive parent has no MappingFEField data. Its diameter is
            // estimated from the active child; all siblings must agree below.
            parent_diameter = 2. * diameter;
          if (distance > 1.1 * half_width ||
              parent_diameter <= .9 * target_diameter)
            // All siblings must agree to coarsen. The margins avoid oscillation
            // at the band boundary and at the target diameter.
            criterion = -1.f;
        }
      }
  }

  template void
  compute_interface_band_criterion<false>(const DoFHandler<2> &,
                                          const Mapping<2> &,
                                          const LA::ParVectorType &,
                                          const unsigned int,
                                          const double,
                                          const double,
                                          Vector<float> &);
  template void
  compute_interface_band_criterion<true>(const DoFHandler<2> &,
                                         const Mapping<2> &,
                                         const LA::ParVectorType &,
                                         const unsigned int,
                                         const double,
                                         const double,
                                         Vector<float> &);

  template <int dim>
  void adapt_with_mmg(const ParameterReader<dim> &param,
                      const MetricField<dim>     &metric_field,
                      const std::string          &adapt_directory,
                      const std::string          &input_meshfile,
                      const std::string          &output_meshfile,
                      const unsigned int          interval_index)
  {
#if defined(FEZ_WITH_MMG)
    // MMG is serial, so adaptation is performed from the root process
    // Maybe look into using ParMMG, but it seems to be no longer in development

    // Gather the metrics to the root process
    const auto gathered_metrics = metric_field.gather_metrics();

    const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(metric_field.get_mpi_communicator());

    if (mpi_rank == 0)
    {
      if (param.mesh.adaptation.verbosity == Parameters::Verbosity::verbose)
      {
        std::cout << std::endl;
        if (param.time_integration.is_steady())
          std::cout << "-- Adapting the mesh with Gmsh and MMG..." << std::endl;
        else
          std::cout
            << "-- Adapting the mesh with Gmsh and MMG for time interval "
            << interval_index << "..." << std::endl;
        std::cout << "\tCurrent mesh file                 : " << input_meshfile
                  << std::endl;
        std::cout << "\tTarget mesh file after adaptation : " << output_meshfile
                  << std::endl;
      }

      const std::string current_mesh_in_msh2 = adapt_directory + "to.msh2";
      const std::string current_mesh_in_medit =
        adapt_directory + "current.mesh";
      const std::string current_sizefield_file =
        adapt_directory + "current_sizefield.sol";

#  if defined(DEAL_II_GMSH_WITH_API)

      // Starting with version 9.8.0, deal.II initializes/finalizes Gmsh in the
      // MPI_InitFinalize call. If that's the case, we don't need to do it here.
      const bool gmsh_initialized_by_dealii = gmsh::isInitialized();
      if (!gmsh_initialized_by_dealii)
        gmsh::initialize();

      /**
       * There seems to be a bug when using deal.II's write_msh with the Gmsh
       * API, when the deal.II mesh is created wiht colorize=true and has
       * physical entities. Until it's figured out, start from a .msh mesh
       * when enabling adaptivity.
       */
      bool use_deal_ii_mesh = param.mesh.deal_ii_preset_mesh != "none";
      AssertThrow(!use_deal_ii_mesh,
                  ExcMessage(
                    "Temporary: anisotropic mesh adaptation is only available "
                    "when starting from a Gmsh .msh mesh file."));

      // Write the current mesh to msh2 format using the Gmsh API
      // (MMG only takes .msh format 2.2 as input)
      gmsh::option::setNumber("General.Verbosity", 2);
      gmsh::open(input_meshfile);

      // MMG does not preserve the names of the physical entities after
      // remeshing, so save the physical entities of the current mesh.
      // FIXME: Would use a map, but for some very weird reason, declaring the
      // description as a map causes a segfault, seemingly from Gmsh. Very odd.
      // std::map<std::pair<int, int>, std::string> my_description;
      std::vector<std::pair<std::pair<int, int>, std::string>> my_description;
      {
        gmsh::vectorpair physical_groups;
        gmsh::model::getPhysicalGroups(physical_groups);
        for (const auto &dimtag : physical_groups)
        {
          std::string physical_entity_name;
          gmsh::model::getPhysicalName(dimtag.first,
                                       dimtag.second,
                                       physical_entity_name);
          // my_description.insert(std::make_pair(dimtag,
          // physical_entity_name));
          my_description.push_back(
            std::make_pair(dimtag, physical_entity_name));
        }
      }

      gmsh::write(current_mesh_in_msh2);
      gmsh::clear();
#  else
      AssertThrow(
        false,
        ExcMessage("Gmsh is required to perform anisotropic mesh adaptation."));
#  endif

      // Initialize the MMG5 mesh and metric structures
      MMG5_pMesh mmgMesh = NULL;
      MMG5_pSol  mmgSol  = NULL;
      int        ier;

      // Get the min and max mesh size from all metric fields
      double min_meshsize = param.metric_fields[0].min_meshsize;
      double max_meshsize = param.metric_fields[0].max_meshsize;
      for (unsigned int i = 1; i < param.metric_fields.size(); ++i)
      {
        // Take the highest min size and the lowest max size
        min_meshsize =
          std::max(min_meshsize, param.metric_fields[i].min_meshsize);
        max_meshsize =
          std::min(max_meshsize, param.metric_fields[i].max_meshsize);
      }

      if constexpr (dim == 2)
      {
        MMG2D_Init_mesh(MMG5_ARG_start,
                        MMG5_ARG_ppMesh,
                        &mmgMesh,
                        MMG5_ARG_ppMet,
                        &mmgSol,
                        MMG5_ARG_end);

        Assert(mmgMesh->np == 0, ExcInternalError());
        Assert(mmgSol->np == 0, ExcInternalError());

        /* Set MMG Verbosity before everything else */
        ier = MMG2D_Set_iparameter(mmgMesh,
                                   mmgSol,
                                   MMG2D_IPARAM_verbose,
                                   param.mesh.adaptation.metric.mmg_verbosity);

        // Load the 2D mesh
        ier = MMG2D_loadMshMesh(mmgMesh, mmgSol, current_mesh_in_msh2.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_loadMshMesh"));

        Assert(static_cast<unsigned int>(mmgMesh->np) ==
                 metric_field.get_n_total_owned_vertices(),
               ExcInternalError());

        // Write the tensor-valued MMG size field from the metric field
        metric_field.set_mmg_solution(gathered_metrics, mmgMesh, mmgSol);

        Assert(mmgMesh->np == mmgSol->np, ExcInternalError());

        // Save current mesh (MEDIT format) and size field
        ier = MMG2D_saveMesh(mmgMesh, current_mesh_in_medit.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_saveMesh"));
        ier = MMG2D_saveSol(mmgMesh, mmgSol, current_sizefield_file.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_saveSol"));

        /* Maximal mesh size (default FLT_MAX)*/
        ier = MMG2D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG2D_DPARAM_hmax,
                                   max_meshsize);
        AssertThrow(
          ier == 1,
          ExcMessage(
            "Error in MMG2D_Set_dparameter when setting max mesh size"));

        /* Minimal mesh size (default 0)*/
        ier = MMG2D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG2D_DPARAM_hmin,
                                   min_meshsize);
        AssertThrow(
          ier == 1,
          ExcMessage(
            "Error in MMG2D_Set_dparameter when setting min mesh size"));

        /* Gradation control*/
        // Disable gradation on MMG's side completely.
        // Gradation is expected to be applied to the metric field on our end.
        ier = MMG2D_Set_dparameter(mmgMesh, mmgSol, MMG2D_DPARAM_hgrad, -1.);
        AssertThrow(ier == 1,
                    ExcMessage(
                      "Error in MMG2D_Set_dparameter when setting gradation"));

        // Adapt the mesh!
        ier = MMG2D_mmg2dlib(mmgMesh, mmgSol);

        if (ier == MMG5_STRONGFAILURE)
          AssertThrow(
            false, ExcMessage("BAD ENDING OF MMG2DLIB: UNABLE TO SAVE MESH\n"));
        else if (ier == MMG5_LOWFAILURE)
          AssertThrow(false, ExcMessage("BAD ENDING OF MMG2DLIB\n"));

        if (param.mesh.adaptation.verbosity == Parameters::Verbosity::verbose)
        {
          std::cout << "\tSuccessfully adapted the mesh" << std::endl;
          std::cout << "\tNumber of mesh vertices after adaptation : "
                    << mmgMesh->np << std::endl;
          std::cout << std::endl;
        }

        // Write the adapted mesh and size field
        ier = MMG2D_saveMshMesh(mmgMesh, mmgSol, output_meshfile.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_saveMshMesh"));
        ier = MMG2D_saveSol(mmgMesh, mmgSol, output_meshfile.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_saveSol"));

        // Free the MMG structures
        MMG2D_Free_all(MMG5_ARG_start,
                       MMG5_ARG_ppMesh,
                       &mmgMesh,
                       MMG5_ARG_ppMet,
                       &mmgSol,
                       MMG5_ARG_end);
      }
      else
      {
        MMG3D_Init_mesh(MMG5_ARG_start,
                        MMG5_ARG_ppMesh,
                        &mmgMesh,
                        MMG5_ARG_ppMet,
                        &mmgSol,
                        MMG5_ARG_end);

        Assert(mmgMesh->np == 0, ExcInternalError());
        Assert(mmgSol->np == 0, ExcInternalError());

        /* Set MMG Verbosity before everything else */
        ier = MMG3D_Set_iparameter(mmgMesh,
                                   mmgSol,
                                   MMG3D_IPARAM_verbose,
                                   param.mesh.adaptation.metric.mmg_verbosity);

        // Load the 3D mesh
        ier = MMG3D_loadMshMesh(mmgMesh, mmgSol, current_mesh_in_msh2.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_loadMshMesh"));

        Assert(static_cast<unsigned int>(mmgMesh->np) ==
                 metric_field.get_n_total_owned_vertices(),
               ExcInternalError());

        // Write the tensor-valued MMG size field from the metric field
        metric_field.set_mmg_solution(gathered_metrics, mmgMesh, mmgSol);

        Assert(mmgMesh->np == mmgSol->np, ExcInternalError());

        // Save initial mesh (MEDIT format) and size field
        ier = MMG3D_saveMesh(mmgMesh, current_mesh_in_medit.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_saveMesh"));
        ier = MMG3D_saveSol(mmgMesh, mmgSol, current_sizefield_file.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_saveSol"));

        /* Maximal mesh size */
        ier = MMG3D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG3D_DPARAM_hmax,
                                   max_meshsize);
        AssertThrow(
          ier == 1,
          ExcMessage(
            "Error in MMG3D_Set_dparameter when setting max mesh size"));

        /* Minimal mesh size */
        ier = MMG3D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG3D_DPARAM_hmin,
                                   min_meshsize);
        AssertThrow(
          ier == 1,
          ExcMessage(
            "Error in MMG3D_Set_dparameter when setting min mesh size"));

        /* Gradation control*/
        // Disable gradation on MMG's side completely.
        // Gradation is expected to be applied to the metric field on our end.
        ier = MMG3D_Set_dparameter(mmgMesh, mmgSol, MMG3D_DPARAM_hgrad, -1.);
        AssertThrow(ier == 1,
                    ExcMessage(
                      "Error in MMG3D_Set_dparameter when setting gradation"));

        // Adapt the mesh!
        ier = MMG3D_mmg3dlib(mmgMesh, mmgSol);

        if (ier == MMG5_STRONGFAILURE)
          AssertThrow(
            false, ExcMessage("BAD ENDING OF MMG3DLIB: UNABLE TO SAVE MESH\n"));
        else if (ier == MMG5_LOWFAILURE)
          AssertThrow(false, ExcMessage("BAD ENDING OF MMG3DLIB\n"));

        if (param.mesh.adaptation.verbosity == Parameters::Verbosity::verbose)
        {
          std::cout << "\tSuccessfully adapted the mesh" << output_meshfile
                    << std::endl;
          std::cout << "\tNumber of mesh vertices after adaptation : "
                    << mmgMesh->np << std::endl;
        }

        // Write the adapted mesh and size field
        ier = MMG3D_saveMshMesh(mmgMesh, mmgSol, output_meshfile.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_saveMshMesh"));
        ier = MMG3D_saveSol(mmgMesh, mmgSol, output_meshfile.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_saveSol"));

        // Free the MMG structures
        MMG3D_Free_all(MMG5_ARG_start,
                       MMG5_ARG_ppMesh,
                       &mmgMesh,
                       MMG5_ARG_ppMet,
                       &mmgSol,
                       MMG5_ARG_end);
      }

#  if defined(DEAL_II_GMSH_WITH_API)
      // MMG does not save the names of the physical entities, so re-assign them
      // here based on the saved my_description.
      gmsh::option::setNumber("General.Verbosity", 2); // Errors and warnings
      gmsh::open(output_meshfile);

      gmsh::vectorpair physical_groups;
      gmsh::model::getPhysicalGroups(physical_groups);

      for (const auto &[dimtag, name] : my_description)
      {
        const int entity_dim = dimtag.first;
        const int tag        = dimtag.second;

        // MMG seems to randomly add 0-dimensional physical points, which
        // we don't care for. They may or may not carry over remeshing steps,
        // but we don't care if they don't, so simply don't check those.
        // Alternatively, we could remove all physical entities then re-add
        // the 1+ dimensional ones.
        if (entity_dim > 0)
        {
          bool entity_found = false;
          for (const auto &[pdim, ptag] : physical_groups)
          {
            if (pdim == entity_dim && ptag == tag)
            {
              gmsh::model::setPhysicalName(entity_dim, tag, name);
              entity_found = true;
            }
          }

          AssertThrow(
            entity_found, ExcMessage(([&]() {
              std::ostringstream message;
              message << "Physical entity with name \"" << name
                      << "\" and (dimension, gmsh tag) = (" << entity_dim
                      << ", " << tag
                      << ") could not be reassigned after mesh adaptation :/";
              return message.str();
            })()));
        }
      }

      gmsh::write(output_meshfile);
      gmsh::clear();

      if (!gmsh_initialized_by_dealii)
        gmsh::finalize();
#  endif
    }

    /**
     * Important: wait for the mesh to be adapted and written to disk. Otherwise
     * the other ranks will read a corrupted mesh file, leading to all sorts of
     * problems.
     */
    MPI_Barrier(metric_field.get_mpi_communicator());

#else
    AssertThrow(false,
                ExcMessage(
                  "MMG is required to perform anisotropic mesh adaptation."));
    (void)param;
    (void)metric_field;
    (void)adapt_directory;
    (void)input_meshfile;
    (void)output_meshfile;
    (void)interval_index;
#endif
  }

  template void adapt_with_mmg(const ParameterReader<2> &,
                               const MetricField<2> &,
                               const std::string &,
                               const std::string &,
                               const std::string &,
                               const unsigned int);
  template void adapt_with_mmg(const ParameterReader<3> &,
                               const MetricField<3> &,
                               const std::string &,
                               const std::string &,
                               const std::string &,
                               const unsigned int);
} // namespace MeshTools
