
#include <deal.II/grid/grid_out.h>
#include <mesh.h>
#include <mesh_adaptation_tools.h>
#include <metric_field.h>
#include <parameter_reader.h>

#if defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#endif

#if defined(FEZ_WITH_MMG)
#  include <mmg/libmmg.h>
#endif

namespace MeshTools
{
  template <int dim>
  void adapt_with_mmg(const ParameterReader<dim> &param,
                      const Triangulation<dim>   &triangulation,
                      const MetricField<dim>     &metric_field)
  {
#if defined(FEZ_WITH_MMG)

    // MMG is serial, so adaptation is performed from the root process
    // Maybe look into using ParMMG, but it seems to be no longer in development

    // Gather the metrics to the root process
    const auto gathered_metrics = metric_field.gather_metrics();

    const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(triangulation.get_mpi_communicator());

    if (mpi_rank == 0)
    {
      const std::string adapt_dir = param.output.output_dir + "adaptation/";

#  if defined(DEAL_II_GMSH_WITH_API)
      std::string current_meshfile = param.mesh.filename;

      bool use_deal_ii_mesh = param.mesh.deal_ii_preset_mesh != "none";

      if (use_deal_ii_mesh)
      {
        /**
         * There seems to be a bug when using deal.II's write_msh with the Gmsh
         * API, when the deal.II mesh is created wiht colorize=true and has
         * physical entities. Until it's figured out, start from a .msh mesh
         * when enabling adaptivity.
         */
        AssertThrow(param.debug.write_dealii_mesh_as_msh,
                    ExcMessage("Temporary: Need to write the serial mesh to "
                               ".msh format for adaptation."));

        current_meshfile = param.output.output_dir + "mesh_from_dealii.msh";
      }

      // Write the current mesh to msh2 format using the Gmsh API
      // (MMG only takes .msh format 2.2 as input)
      gmsh::initialize();
      gmsh::open(current_meshfile);

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

      gmsh::write(adapt_dir + "to.msh2");
      gmsh::clear();
      gmsh::finalize();
#  else
      AssertThrow(
        false,
        ExcMessage("Gmsh is required to perform anisotropic mesh adaptation."));
#  endif

      // Initialize the MMG5 mesh and metric structures
      MMG5_pMesh mmgMesh = NULL;
      MMG5_pSol  mmgSol  = NULL;
      int        ier;

      std::cout << "dealii Mesh has " << triangulation.n_vertices()
                << " vertices " << std::endl;

      const std::string filename                = adapt_dir + "to.msh2";
      const std::string current_mesh_medit_file = adapt_dir + "current.mesh";
      const std::string current_sizefield_file =
        adapt_dir + "current_sizefield.sol";

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

        // Load the 2D mesh
        ier = MMG2D_loadMshMesh(mmgMesh, mmgSol, filename.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_loadMshMesh"));

        Assert(mmgMesh->np == metric_field.get_n_total_owned_vertices(),
               ExcInternalError());

        // Write the tensor-valued MMG size field from the metric field
        metric_field.set_mmg_solution(gathered_metrics, mmgMesh, mmgSol);

        Assert(mmgMesh->np == mmgSol->np, ExcInternalError());

        // Save current mesh (MEDIT format) and size field
        ier = MMG2D_saveMesh(mmgMesh, current_mesh_medit_file.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_saveMesh"));
        ier = MMG2D_saveSol(mmgMesh, mmgSol, current_sizefield_file.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_saveSol"));

        /* Maximal mesh size (default FLT_MAX)*/
        ier = MMG2D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG2D_DPARAM_hmax,
                                   param.metric_fields[0].max_meshsize);
        AssertThrow(
          ier == 1,
          ExcMessage(
            "Error in MMG2D_Set_dparameter when setting max mesh size"));

        /* Minimal mesh size (default 0)*/
        ier = MMG2D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG2D_DPARAM_hmin,
                                   param.metric_fields[0].min_meshsize);
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

        // Load the 3D mesh
        ier = MMG3D_loadMshMesh(mmgMesh, mmgSol, filename.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_loadMshMesh"));

        Assert(mmgMesh->np == metric_field.get_n_total_owned_vertices(),
               ExcInternalError());

        // Write the tensor-valued MMG size field from the metric field
        metric_field.set_mmg_solution(gathered_metrics, mmgMesh, mmgSol);

        Assert(mmgMesh->np == mmgSol->np, ExcInternalError());

        // Save initial mesh (MEDIT format) and size field
        ier = MMG3D_saveMesh(mmgMesh, current_mesh_medit_file.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_saveMesh"));
        ier = MMG3D_saveSol(mmgMesh, mmgSol, current_sizefield_file.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_saveSol"));

        /* Maximal mesh size */
        ier = MMG3D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG3D_DPARAM_hmax,
                                   param.metric_fields[0].max_meshsize);
        AssertThrow(
          ier == 1,
          ExcMessage(
            "Error in MMG3D_Set_dparameter when setting max mesh size"));

        /* Minimal mesh size */
        ier = MMG3D_Set_dparameter(mmgMesh,
                                   mmgSol,
                                   MMG3D_DPARAM_hmin,
                                   param.metric_fields[0].min_meshsize);
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
      }

      // Write the adapted mesh
      std::string filename_out = adapt_dir + "adapted.msh";

      if constexpr (dim == 2)
      {
        ier = MMG2D_saveMshMesh(mmgMesh, mmgSol, filename_out.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_saveMshMesh"));
        ier = MMG2D_saveSol(mmgMesh, mmgSol, filename_out.c_str());
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
        ier = MMG3D_saveMshMesh(mmgMesh, mmgSol, filename_out.c_str());
        AssertThrow(ier == 1, ExcMessage("Error in MMG3D_saveMshMesh"));
        ier = MMG3D_saveSol(mmgMesh, mmgSol, filename_out.c_str());
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
      gmsh::initialize();
      gmsh::open(filename_out);

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

      gmsh::write(filename_out);
      gmsh::clear();
      gmsh::finalize();
#  endif
    }
#else
    AssertThrow(false,
                ExcMessage(
                  "MMG is required to perform anisotropic mesh adaptation."));
    (void)param;
    (void)triangulation;
    (void)metric_field;
#endif
  }



  template void adapt_with_mmg(const ParameterReader<2> &,
                               const Triangulation<2> &,
                               const MetricField<2> &);
  template void adapt_with_mmg(const ParameterReader<3> &,
                               const Triangulation<3> &,
                               const MetricField<3> &);
} // namespace MeshTools
