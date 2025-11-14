
#include <parameter_reader.h>

template <int dim>
void ParameterReader<dim>::check_parameters() const
{
  if (initial_conditions.set_to_mms && !mms_param.enable)
  {
    throw std::runtime_error(
      "The initial conditions should be prescribed by the manufactured "
      "solution, but either no manufactured solution was provided or it was "
      "not enabled.");
  }
  if (!initial_conditions.set_to_mms && mms_param.enable)
  {
    throw std::runtime_error(
      "A manufactured solution is prescribed, but the initial conditions are "
      "not set to be prescribed by this solution. Set \"set to mms = true\" "
      "for the initial conditions.");
  }
}

template class ParameterReader<2>;
template class ParameterReader<3>;
