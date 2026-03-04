
#include <deal.II/grid/grid_out.h>
#include <mesh_adaptation_tools.h>
#include <mmg/libmmg.h>
#include <parameter_reader.h>

#ifdef DEAL_II_GMSH_WITH_API
#  include <gmsh.h>
#endif

template <int dim>
void adapt_mesh_mmg(const ParameterReader<dim> &param,
                    const Triangulation<dim>   &triangulation)
{
  // Start by writing the mesh to a Gmsh format
  // MMG only accepts .msh format 2.2, whereas deal.II only writes to format 4.1
  GridOut grid_out;
  grid_out.write_msh(triangulation,
                     param.output.output_dir + "to_version4.msh");

  gmsh::initialize();
  gmsh::open(param.output.output_dir + "to_version4.msh");
  gmsh::write(param.output.output_dir + "to.msh2");
  gmsh::clear();
  gmsh::finalize();

  MMG5_pMesh  mmgMesh;
  MMG5_pSol   mmgSol;
  int         ier;
  std::string filename, filename_o2d, filename_o3d;
  std::size_t found;

  std::cout << "Mesh has " << triangulation.n_vertices() << " vertices "
            << std::endl;

  fprintf(stdout, "  -- TEST MMGLIB \n");

  /** ================== 2d remeshing using the mmg2d library ========== */
  filename = param.output.output_dir + "to.msh2";

  std::cout << filename << std::endl;

  mmgMesh = NULL;
  mmgSol  = NULL;
  MMG2D_Init_mesh(MMG5_ARG_start,
                  MMG5_ARG_ppMesh,
                  &mmgMesh,
                  MMG5_ARG_ppMet,
                  &mmgSol,
                  MMG5_ARG_end);

  /** 2) Build mesh in MMG5 format */
  /** with MMG2D_loadMesh function */
  ier = MMG2D_loadMshMesh(mmgMesh, mmgSol, filename.c_str());
  AssertThrow(ier == 1, ExcMessage("Error in MMG2D_loadMshMesh"));

  std::cout << "Mesh has " << mmgMesh->xp << " vertices " << std::endl;

  ier = MMG2D_loadSol(mmgMesh, mmgSol, filename.c_str());
  AssertThrow(ier == 1 || ier == 0, ExcMessage("Error in MMG2D_loadSol"));

  /** Manually set of the sol */
  /** a) give info for the sol structure: sol applied on vertex entities,
      number of vertices=4, the sol is scalar*/
  // if ( MMG2D_Set_solSize(mmgMesh, mmgSol, MMG5_Vertex, 4, MMG5_Scalar) != 1 )
  //   exit(EXIT_FAILURE);

  /** b) give solutions values and positions */
  // for(unsigned int k = 1 ; k <= mmgMesh->xp ; k++) {
  //   if ( MMG2D_Set_scalarSol(mmgSol,0.01,k) != 1 ) exit(EXIT_FAILURE);
  // }

  // /** 3) Build sol in MMG5 format */
  // if ( MMG2D_loadSol(mmgMesh, mmgSol,filename.c_str()) != 1 )
  // exit(EXIT_FAILURE);

  /** ------------------------------ STEP  II -------------------------- */
  /** remesh function */
  ier = MMG2D_mmg2dlib(mmgMesh, mmgSol);

  if (ier == MMG5_STRONGFAILURE)
    AssertThrow(false,
                ExcMessage("BAD ENDING OF MMG2DLIB: UNABLE TO SAVE MESH\n"));
  else if (ier == MMG5_LOWFAILURE)
    fprintf(stdout, "BAD ENDING OF MMG2DLIB\n");

  /** ------------------------------ STEP III -------------------------- */
  /** get results */
  /** Two solutions: just use the MMG2D_saveMesh/MMG2D_saveSol functions
      that will write .mesh(b)/.sol formatted files or manually get your
     mesh/sol using the MMG2D_getMesh/MMG2D_getSol functions */

  filename_o2d = param.output.output_dir + "final";

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
}

template void adapt_mesh_mmg(const ParameterReader<2> &,
                             const Triangulation<2> &);
template void adapt_mesh_mmg(const ParameterReader<3> &,
                             const Triangulation<3> &);
