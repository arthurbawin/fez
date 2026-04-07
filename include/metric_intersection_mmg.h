/**
 * This file gathers the functions MMG5_intersecmet22 and MMG5_intersecmet33,
 * from the MMG library.
 *
 * Credit goes to the MMG authors listed in metric_intersection.cpp.
 */

/* =============================================================================
**  This file is part of the mmg software package for the tetrahedral
**  mesh modification.
**  Copyright (c) Bx INP/CNRS/Inria/UBordeaux/UPMC, 2004-
**
**  mmg is free software: you can redistribute it and/or modify it
**  under the terms of the GNU Lesser General Public License as published
**  by the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  mmg is distributed in the hope that it will be useful, but WITHOUT
**  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
**  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
**  License for more details.
**
**  You should have received a copy of the GNU Lesser General Public
**  License and of the GNU General Public License along with mmg (in
**  files COPYING.LESSER and COPYING). If not, see
**  <http://www.gnu.org/licenses/>. Please read their terms carefully and
**  use this copy of the mmg distribution only if you accept them.
** =============================================================================
*/
#ifndef METRIC_INTERSECTION_MMG_H
#define METRIC_INTERSECTION_MMG_H

/**
 * Intersection of the 2x2 metrics m1 and m2.
 *
 * \param hmin minimum prescribed mesh size.
 * \param hmax maximum prescribed mesh size.
 * \param m pointer to a \f$(2x2)\f$ metric.
 * \param n pointer to a \f$(2x2)\f$ metric.
 * \param mr computed \f$(2x2)\f$ metric.
 * \return 0 if fail, 1 otherwise.
 *
 * Compute the intersected (2 x 2) metric from metrics \a m and \a n : take
 * simultaneous reduction, and proceed to truncation in sizes.
 */
int MMG5_intersecmet22(double  hmin,
                       double  hmax,
                       double *m1,
                       double *m2,
                       double *result);

/**
 * Intersection of the 3x3 metrics m1 and m2.
 *
 * \param hmin minimum prescribed mesh size.
 * \param hmax maximum prescribed mesh size.
 * \param m pointer to a \f$(3x3)\f$ metric.
 * \param n pointer to a \f$(3x3)\f$ metric.
 * \param mr computed \f$(3x3)\f$ metric.
 * \return 0 if fail, 1 otherwise.
 *
 * Compute the intersected (3 x 3) metric from metrics \a m and \a n : take
 * simultaneous reduction, and proceed to truncation in sizes.
 */
int MMG5_intersecmet33(double  hmin,
                       double  hmax,
                       double *m1,
                       double *m2,
                       double *result);

#endif
