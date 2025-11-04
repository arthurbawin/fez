#ifndef TYPES_H
#define TYPES_H

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/vector.h>

#if defined(DEAL_II_WITH_PETSC)
using ParVectorType = dealii::LinearAlgebraPETSc::MPI::Vector;
#else
using ParVectorType = dealii::LinearAlgebra::distributed::Vector<double>;
#endif

#endif