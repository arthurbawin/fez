
#include "error_estimation/solution_recovery.h"

namespace ErrorEstimation
{
  using namespace dealii;

  template <int dim>
  Patches<dim>::Patches(const Triangulation<dim> &triangulation,
                        const DoFHandler<dim>    &dof_handler,
                        unsigned int              degree)
    : triangulation(triangulation)
    , dof_handler(dof_handler)
  {
    // Build patches with enough vertices to fit a degree p+1 polynomial
    unsigned int num_required_vertices =
      PolynomialSpace<dim>::dim_polynomial_basis(degree + 1);

    const auto n_vertices = triangulation.n_vertices();

    patches.resize(n_vertices);
    patches_of_vertices.resize(n_vertices);
    num_layers.resize(n_vertices);

    // Add layers of elements until there are at least num_required_vertices
    // vertices in the patch
    for (unsigned int i = 0; i < n_vertices; ++i)
    {
      do
      {
        addLayer(i);
      }
      while (patches_of_vertices[i].size() < num_required_vertices);

      AssertThrow(patches_of_vertices[i].size() >= num_required_vertices,
                  ExcMessage("foo2"));
    }

    this->compute_scalings();
  }

  template <int dim>
  void Patches<dim>::addLayer(unsigned int vertex_index)
  {
    auto         &patch          = patches[vertex_index];
    auto         &patch_vertices = patches_of_vertices[vertex_index];
    unsigned int &layer          = num_layers[vertex_index];

    std::set<typename DoFHandler<dim>::active_cell_iterator> new_cells;
    std::set<unsigned int>                                   new_vertices;

    if (layer == 0)
    {
      // First layer: cells containing the vertex directly
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int n_vertices_per_cell = cell->n_vertices();
        for (unsigned int i = 0; i < n_vertices_per_cell; ++i)
        {
          if (cell->vertex_index(i) == vertex_index)
          {
            new_cells.insert(cell);
            break;
          }
        }
      }
    }
    else
    {
      // Collect all unique vertices from the previous layer's cells
      std::set<unsigned int> adjacent_vertices;

      for (const auto &cell : patch)
      {
        const unsigned int n_vertices_per_cell = cell->n_vertices();
        for (unsigned int i = 0; i < n_vertices_per_cell; ++i)
        {
          adjacent_vertices.insert(cell->vertex_index(i));
        }
      }

      // Now collect all cells touching any of these adjacent vertices
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int n_vertices_per_cell = cell->n_vertices();
        for (unsigned int i = 0; i < n_vertices_per_cell; ++i)
        {
          if (adjacent_vertices.count(cell->vertex_index(i)) > 0)
          {
            if (patch.count(cell) == 0)
              new_cells.insert(cell);
            break;
          }
        }
      }
    }

    // Add new cells to the patch
    patch.insert(new_cells.begin(), new_cells.end());

    // Add cell vertices to patch of vertices
    for (const auto &new_cell : new_cells)
    {
      const unsigned int n_vertices_per_cell = new_cell->n_vertices();
      for (unsigned int i = 0; i < n_vertices_per_cell; ++i)
      {
        new_vertices.insert(new_cell->vertex_index(i));
      }
    }

    // patch_vertices.reserve(patch_vertices.size() + new_vertices.size());
    // patch_vertices.assign(new_vertices.begin(), new_vertices.end());
    patch_vertices.insert(new_vertices.begin(), new_vertices.end());

    ++layer;
  }

  template <int dim>
  void Patches<dim>::compute_scalings()
  {
    const auto                     n_vertices = triangulation.n_vertices();
    const std::vector<Point<dim>> &vertices   = triangulation.get_vertices();

    scalings.resize(n_vertices, Point<dim>());

    for (unsigned int i = 0; i < n_vertices; ++i)
    {
      const Point<dim> &vi                = vertices[i];
      Point<dim>       &scaling           = scalings[i];
      const auto       &patch_of_vertices = patches_of_vertices[i];

      for (const auto &ind_j : patch_of_vertices)
      {
        const Point<dim> &vj = vertices[ind_j];
        for (unsigned int k = 0; k < dim; ++k)
          scaling[k] = std::max(scaling[k], std::abs(vi[k] - vj[k]));
      }
    }
  }

  template <int dim>
  void Patches<dim>::increase_patch_size(unsigned int vertex_index)
  {
    this->addLayer(vertex_index);

    // Update scaling
    Point<dim>                    &scaling  = scalings[vertex_index];
    const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

    const Point<dim> &vi                = vertices[vertex_index];
    const auto       &patch_of_vertices = patches_of_vertices[vertex_index];

    for (const auto &ind_j : patch_of_vertices)
    {
      const Point<dim> &vj = vertices[ind_j];
      for (unsigned int k = 0; k < dim; ++k)
        scaling[k] = std::max(scaling[k], std::abs(vi[k] - vj[k]));
    }
  }

  template <int dim>
  void Patches<dim>::write_patch_to_pos(const unsigned int i,
                                        const unsigned int posTag) const
  {
    Assert(dim == 2, ExcNotImplemented());

    const auto &cell_set = patches[i];
    const auto &scaling  = scalings[i];

    // Write .pos file
    std::ofstream out("patch_vertex_" + std::to_string(i) + "_" +
                      std::to_string(posTag) + ".pos");
    out << "View \"Patch " << i << "\" {\n";

    // The elements of the patch
    for (const auto &cell : cell_set)
    {
      Assert(cell->n_vertices() == 3, ExcInternalError());

      out << "ST(";
      for (unsigned int j = 0; j < 3; ++j)
      {
        const Point<dim> &pt = cell->vertex(j);
        out << pt[0] << "," << pt[1] << ",0";
        if (j < 2)
          out << ",";
      }
      out << "){1., 1., 1.};\n";
    }

    // The bounding box
    const Point<dim> &pt   = triangulation.get_vertices()[i];
    const double      xmin = pt[0] - scaling[0];
    const double      xmax = pt[0] + scaling[0];
    const double      ymin = pt[1] - scaling[1];
    const double      ymax = pt[1] + scaling[1];
    out << "SL(" << xmin << "," << ymin << ",0.," << xmax << "," << ymin
        << ",0.){1., 1.};\n";
    out << "SL(" << xmax << "," << ymin << ",0.," << xmax << "," << ymax
        << ",0.){1., 1.};\n";
    out << "SL(" << xmax << "," << ymax << ",0.," << xmin << "," << ymax
        << ",0.){1., 1.};\n";
    out << "SL(" << xmin << "," << ymax << ",0.," << xmin << "," << ymin
        << ",0.){1., 1.};\n";

    out << "};\n";
    out.close();
  }

  // Explicit instantiations
  template class Patches<2>;
  template class Patches<3>;

} // namespace ErrorEstimation
