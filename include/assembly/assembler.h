#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include <deal.II/base/observer_pointer.h>

/**
 * A namespace gathering the assembly routines, each associated with a volume
 * or boundary weak form.
 */
namespace Assembly
{
  using namespace dealii;

  /**
   * Base class for an object whose job is to assemble equations on an element,
   * that is, to increment the local matrix and right-hand side vector.
   *
   * This design is borrowed from the chaos-polymtl/lethe project.
   */
  template <typename ScratchData, typename CopyData>
  class AssemblerBase : public EnableObserverPointer
  {
  public:
    /**
     * Assemble the local matrix on an element, and store the result in the
     * @p copy_data. The @p scratch_data must have been reinit'ed on that
     * element before calling this function.
     */
    virtual void assemble_matrix(const ScratchData &scratch_data,
                                 CopyData          &copy_data) const = 0;

    /**
     * Assemble the local right-hand side on an element, and store the result in
     * the @p copy_data. The @p scratch_data must have been reinit'ed on that
     * element before calling this function.
     */
    virtual void assemble_rhs(const ScratchData &scratch_data,
                              CopyData          &copy_data) const = 0;
  };
} // namespace Assembly

#endif
