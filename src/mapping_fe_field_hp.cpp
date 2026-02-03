
// #include <mapping_fe_field_hp.templates.h>
#include <deal.II/lac/vector.h>
#include <mapping_fe_field_hp2.templates.h>
#include <types.h>

DEAL_II_NAMESPACE_OPEN

// template class MappingFEFieldHp<2, 2, Vector<double>>;
// template class MappingFEFieldHp<2, 2, LA::ParVectorType>;
// template class MappingFEFieldHp<3, 3, Vector<double>>;
// template class MappingFEFieldHp<3, 3, LA::ParVectorType>;

template class MappingFEFieldHp2<2, 2, Vector<double>>;
template class MappingFEFieldHp2<2, 2, LA::ParVectorType>;
template class MappingFEFieldHp2<3, 3, Vector<double>>;
template class MappingFEFieldHp2<3, 3, LA::ParVectorType>;

DEAL_II_NAMESPACE_CLOSE