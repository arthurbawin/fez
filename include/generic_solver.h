#ifndef GENERIC_SOLVER_H
#define GENERIC_SOLVER_H


using namespace dealii;

/**
 * Abstract base class for a generic solver with the following common members:
 * 
 * - nonlinear solver
 * - simulation parameters
 * 
 * A generic solver is dimension agnostic, and thus does not contain e.g. the mesh.
 */
template <typename VectorType>
class GenericSolver
{
public:
	GenericSolver(){};

public:
	virtual ~GenericSolver();

public:
}

#endif