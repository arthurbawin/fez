
#include <deal.II/grid/grid_out.h>
#include <mesh_adaptation_tools.h>
#include <metric_field.h>
#include <parameter_reader.h>

#if defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#endif

#if defined(FEZ_WITH_MMG)
#  include <mmg/libmmg.h>
#endif

namespace MeshAdaptation
{
  template <int dim>
  void adapt_with_mmg(const ParameterReader<dim> &param,
                      const Triangulation<dim>   &triangulation,
                      const MetricField<dim>     &metric_field)
  {
#if defined(FEZ_WITH_MMG)
    const std::string adapt_dir = param.output.output_dir + "adaptation/";

#  if defined(DEAL_II_GMSH_WITH_API)
    // Start by writing the mesh to a Gmsh format
    // MMG only accepts .msh format 2.2, whereas deal.II only writes to
    // format 4.1
    GridOut grid_out;
    grid_out.write_msh(triangulation, adapt_dir + "to_version4.msh");

    gmsh::initialize();
    gmsh::open(adapt_dir + "to_version4.msh");
    gmsh::write(adapt_dir + "to.msh2");
    gmsh::clear();
    gmsh::finalize();
#  else
    AssertThrow(false,
                ExcMessage(
                  "Gmsh is required to perform anisotropic mesh adaptation."));
#  endif

    // Initialize the MMG5 mesh and metric structures
    MMG5_pMesh  mmgMesh;
    MMG5_pSol   mmgSol;
    int         ier;
    std::string filename, filename_o2d;
    std::size_t found;

    std::cout << "dealii Mesh has " << triangulation.n_vertices()
              << " vertices " << std::endl;

    fprintf(stdout, "  -- TEST MMGLIB \n");

    filename = adapt_dir + "to.msh2";

    std::cout << filename << std::endl;

    mmgMesh = NULL;
    mmgSol  = NULL;
    MMG2D_Init_mesh(MMG5_ARG_start,
                    MMG5_ARG_ppMesh,
                    &mmgMesh,
                    MMG5_ARG_ppMet,
                    &mmgSol,
                    MMG5_ARG_end);

    std::cout << "Mesh before load has " << mmgMesh->np << " vertices "
              << std::endl;

    /** 2) Build mesh in MMG5 format */
    /** with MMG2D_loadMesh function */
    ier = MMG2D_loadMshMesh(mmgMesh, mmgSol, filename.c_str());
    AssertThrow(ier == 1, ExcMessage("Error in MMG2D_loadMshMesh"));

    std::cout << "Mesh after load  has " << mmgMesh->np << " vertices "
              << std::endl;

    ier = MMG2D_loadSol(mmgMesh, mmgSol, filename.c_str());
    AssertThrow(ier == 1 || ier == 0, ExcMessage("Error in MMG2D_loadSol"));

    metric_field.set_mmg_solution(mmgMesh, mmgSol);

    std::cout << "Mesh has " << mmgMesh->np << " vertices " << std::endl;
    std::cout << "Sol  has " << mmgSol->np << " vertices " << std::endl;

    // Save initial mesh and size field
    if (MMG2D_saveMesh(mmgMesh, "test.mesh") != 1)
      exit(EXIT_FAILURE);
    if (MMG2D_saveSol(mmgMesh, mmgSol, "test.sol") != 1)
      exit(EXIT_FAILURE);

    // Adapt the mesh
    ier = MMG2D_mmg2dlib(mmgMesh, mmgSol);

    if (ier == MMG5_STRONGFAILURE)
      AssertThrow(false,
                  ExcMessage("BAD ENDING OF MMG2DLIB: UNABLE TO SAVE MESH\n"));
    else if (ier == MMG5_LOWFAILURE)
      AssertThrow(false, ExcMessage("BAD ENDING OF MMG2DLIB\n"));

    /** ------------------------------ STEP III -------------------------- */
    /** get results */
    /** Two solutions: just use the MMG2D_saveMesh/MMG2D_saveSol functions
        that will write .mesh(b)/.sol formatted files or manually get your
       mesh/sol using the MMG2D_getMesh/MMG2D_getSol functions */

    filename_o2d = adapt_dir + "final";

    found = filename_o2d.find(".mesh");
    if (found == std::string::npos)
    {
      found = filename_o2d.find(".msh");
    }

    if (found != std::string::npos)
    {
      filename_o2d.replace(found, 1, "\0");
    }
    filename_o2d += ".2d";
    filename_o2d += ".msh";
    /** 1) Automatically save the mesh */
    /*save result*/
    // if ( MMG2D_saveMesh(mmgMesh,filename_o2d.c_str()) != 1 )
    // exit(EXIT_FAILURE);
    if (MMG2D_saveMshMesh(mmgMesh, mmgSol, filename_o2d.c_str()) != 1)
      exit(EXIT_FAILURE);
    /*save metric*/
    // if ( MMG2D_saveSol(mmgMesh,mmgSol,filename_o2d.c_str()) != 1 )
    // exit(EXIT_FAILURE);

    /** 2) Free the MMG2D structures */
    MMG2D_Free_all(MMG5_ARG_start,
                   MMG5_ARG_ppMesh,
                   &mmgMesh,
                   MMG5_ARG_ppMet,
                   &mmgSol,
                   MMG5_ARG_end);
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
} // namespace MeshAdaptation
