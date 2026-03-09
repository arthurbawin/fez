/**
 * This file gathers the functions MMG5_intersecmet22 and MMG5_intersecmet33,
 * from the MMG library, as well as their dependencies. This includes functions
 * from the files:
 *
 * - mmg/src/common/mettools.c
 * - mmg/src/common/eigenv.c
 * - mmg/src/common/mmgcommon_private.h
 * - mmg/src/common/tools.c
 * - mmg/src/common/inlined_functions_private.h
 * - mmg/src/common/anisosiz.c
 *
 * Credit goes to the MMG authors listed below.
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

/**
 * \author Charles Dapogny (UPMC)
 * \author Cécile Dobrzynski (Bx INP/Inria/UBordeaux)
 * \author Pascal Frey (UPMC)
 * \author Algiane Froehly (Inria/UBordeaux)
 * \version 5
 * \copyright GNU Lesser General Public License.
 */

#include <assert.h>
#include <math.h>
#include <metric_intersection_mmg.h>
#include <stdio.h>
#include <string.h>

#define MG_EIGENV_EPS27 1.e-27
#define MG_EIGENV_EPS13 1.e-13
#define MG_EIGENV_EPS10 1.e-10
#define MG_EIGENV_EPS5e6 5.e-06
#define MG_EIGENV_EPS6 1.e-06
#define MG_EIGENV_EPS2e6 2.e-06
#define MG_EIGENV_EPS5 1.e-05
#define MAXTOU 50

#define MMG5_EPSOK 1.e-15

#define MMG5_EPS 1.e-06
#define MMG5_EPSD 1.e-30
#define MMG5_EPSD2 1.0e-200

#define MG_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MG_MIN(a, b) (((a) < (b)) ? (a) : (b))

/**
 * \brief Identity matrix.
 */
static double Id[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

/**
 * \param m terms of symetric matrix \f$2x2\f$.
 * \param lambda eigenvalues of \a m.
 * \param vp eigenvectors of \a m.
 * \return order of the eigenvalues.
 *
 * Compute eigenelements of a symetric matrix m. Eigenvectors are orthogonal.
 *
 */
inline int MMG5_eigensym(double m[3], double lambda[2], double vp[2][2])
{
  double     sqDelta, dd, trm, vnorm, maxm, a11, a12, a22;
  static int ddebug = 0;

  lambda[0] = lambda[1] = 0.;
  vp[0][0] = vp[0][1] = vp[1][0] = vp[1][1] = 0.;

  maxm = fabs(m[0]);
  maxm = fabs(m[1]) > maxm ? fabs(m[1]) : maxm;
  maxm = fabs(m[2]) > maxm ? fabs(m[2]) : maxm;

  if (maxm < MG_EIGENV_EPS13)
  {
    if (ddebug)
      printf("  ## Warning:%s: Quasi-null matrix.", __func__);
    maxm = 1.;
  }

  /* normalize matrix */
  dd  = 1.0 / maxm;
  a11 = m[0] * dd;
  a12 = m[1] * dd;
  a22 = m[2] * dd;

  dd        = a11 - a22;
  trm       = a11 + a22;
  sqDelta   = sqrt(dd * dd + 4.0 * a12 * a12);
  lambda[0] = 0.5 * (trm - sqDelta);

  /* Case when m = lambda[0]*I */
  if (sqDelta < MMG5_EPS)
  {
    lambda[0] *= maxm;
    lambda[1] = lambda[0];
    vp[0][0]  = 1.0;
    vp[0][1]  = 0.0;

    vp[1][0] = 0.0;
    vp[1][1] = 1.0;
    return 2;
  }
  /* Remark: the computation of an independent basis of eigenvectors fail if the
   * matrix is diagonal (we find twice the same eigenvector) */
  vp[0][0] = a12;
  vp[0][1] = (lambda[0] - a11);
  vnorm    = sqrt(vp[0][0] * vp[0][0] + vp[0][1] * vp[0][1]);

  if (vnorm < MMG5_EPS)
  {
    vp[0][0] = (lambda[0] - a22);
    vp[0][1] = a12;
    vnorm    = sqrt(vp[0][0] * vp[0][0] + vp[0][1] * vp[0][1]);
  }
  assert(vnorm > MMG5_EPSD);

  vnorm = 1.0 / vnorm;
  vp[0][0] *= vnorm;
  vp[0][1] *= vnorm;

  vp[1][0] = -vp[0][1];
  vp[1][1] = vp[0][0];

  lambda[1] = a11 * vp[1][0] * vp[1][0] + 2.0 * a12 * vp[1][0] * vp[1][1] +
              a22 * vp[1][1] * vp[1][1];

  lambda[0] *= maxm;
  lambda[1] *= maxm;

  /* Check orthogonality of eigenvectors. If they are not, we probably miss the
   * dectection of a diagonal matrix. */
  assert(fabs(vp[0][0] * vp[1][0] + vp[0][1] * vp[1][1]) <= MG_EIGENV_EPS6);

  return 1;
}

/**
 * \brief Find eigenvalues and vectors of a 2x2 matrix.
 * \param symmat 0 if matrix is not symetric, 1 otherwise.
 * \param mat pointer to the matrix.
 * \param lambda eigenvalues.
 * \param v eigenvectors.
 *
 * \return order of eigenvalues (1,2) or 0 if failed.
 *
 * \remark the i^{th} eigenvector is stored in v[i][.].
 *
 */
int MMG5_eigenv2d(int symmat, double *mat, double lambda[2], double vp[2][2])
{
  double        dd, sqDelta, trmat, vnorm;
  static int8_t mmgWarn0 = 0;

  /* wrapper function if symmetric matrix */
  if (symmat)
    return MMG5_eigensym(mat, lambda, vp);


  dd      = mat[0] - mat[3];
  sqDelta = sqrt(fabs(dd * dd + 4.0 * mat[1] * mat[2]));
  trmat   = mat[0] + mat[3];

  lambda[0] = 0.5 * (trmat - sqDelta);
  if (lambda[0] < 0.0)
  {
    if (!mmgWarn0)
    {
      mmgWarn0 = 1;
      fprintf(stderr,
              "\n  ## Warning: %s: at least 1 metric with a "
              "negative eigenvalue: %f \n",
              __func__,
              lambda[0]);
    }
    return 0;
  }

  /* First case : matrices m and n are homothetic: n = lambda0*m */
  if (sqDelta < MMG5_EPS)
  {
    /* only one eigenvalue with degree 2 */
    return 2;
  }
  /* Second case: both eigenvalues of mat are distinct ; theory says qf
   associated to m and n are diagonalizable in basis (vp[0], vp[1]) - the
   coreduction basis */
  else
  {
    lambda[1] = 0.5 * (trmat + sqDelta);
    assert(lambda[1] >= 0.0);

    vp[0][0] = mat[1];
    vp[0][1] = (lambda[0] - mat[0]);
    vnorm    = sqrt(vp[0][0] * vp[0][0] + vp[0][1] * vp[0][1]);

    if (vnorm < MMG5_EPS)
    {
      vp[0][0] = (lambda[0] - mat[3]);
      vp[0][1] = mat[2];
      vnorm    = sqrt(vp[0][0] * vp[0][0] + vp[0][1] * vp[0][1]);
    }

    vnorm = 1.0 / vnorm;
    vp[0][0] *= vnorm;
    vp[0][1] *= vnorm;

    vp[1][0] = mat[1];
    vp[1][1] = (lambda[1] - mat[0]);
    vnorm    = sqrt(vp[1][0] * vp[1][0] + vp[1][1] * vp[1][1]);

    if (vnorm < MMG5_EPS)
    {
      vp[1][0] = (lambda[1] - mat[3]);
      vp[1][1] = mat[2];
      vnorm    = sqrt(vp[1][0] * vp[1][0] + vp[1][1] * vp[1][1]);
    }

    vnorm = 1.0 / vnorm;
    vp[1][0] *= vnorm;
    vp[1][1] *= vnorm;

    /* two distinct eigenvalues with degree 1 */
    return 1;
  }
}

/**
 * \param mesh pointer to the mesh structure.
 * \param m pointer to a \f$(2x2)\f$ metric.
 * \param n pointer to a \f$(2x2)\f$ metric.
 * \param mr computed \f$(2x2)\f$ metric.
 * \return 0 if fail, 1 otherwise.
 *
 * Compute the intersected (2 x 2) metric from metrics \a m and \a n : take
 * simultaneous reduction, and proceed to truncation in sizes.
 *
 */
int MMG5_intersecmet22(double  hmin,
                       double  hmax,
                       double *m,
                       double *n,
                       double *mr)
{
  double        det, imn[4], lambda[2], vp[2][2], dm[2], dn[2], d0, d1, ip[4];
  double        isqhmin, isqhmax;
  static int8_t mmgWarn0 = 0;
  int           order;

  isqhmin = 1.0 / (hmin * hmin);
  isqhmax = 1.0 / (hmax * hmax);

  /* Compute imn = M^{-1}N */
  det = m[0] * m[2] - m[1] * m[1];
  if (fabs(det) < MMG5_EPS * MMG5_EPS)
  {
    if (!mmgWarn0)
    {
      fprintf(stderr,
              "\n  ## Warning: %s: null metric det : %E \n",
              __func__,
              det);
      mmgWarn0 = 1;
    }
    return 0;
  }
  det = 1.0 / det;

  imn[0] = det * (m[2] * n[0] - m[1] * n[1]);
  imn[1] = det * (m[2] * n[1] - m[1] * n[2]);
  imn[2] = det * (-m[1] * n[0] + m[0] * n[1]);
  imn[3] = det * (-m[1] * n[1] + m[0] * n[2]);

  /* Find eigenvalues of imn */
  order = MMG5_eigenv2d(0, imn, lambda, vp);

  if (!order)
  {
    if (!mmgWarn0)
    {
      mmgWarn0 = 1;
      fprintf(stderr,
              "\n  ## Warning: %s: at least 1 failing"
              " simultaneous reduction.\n",
              __func__);
    }
    return 0;
  }

  /* First case : matrices m and n are homothetic : n = lambda0*m */
  if (order == 2)
  {
    /* Diagonalize m and truncate eigenvalues : trimn, det, etc... are reused */
    if (fabs(m[1]) < MMG5_EPS)
    {
      dm[0]    = m[0];
      dm[1]    = m[2];
      vp[0][0] = 1;
      vp[0][1] = 0;
      vp[1][0] = 0;
      vp[1][1] = 1;
    }
    else
    {
      MMG5_eigensym(m, dm, vp);
    }
    /* Eigenvalues of the resulting matrix*/
    dn[0] = MG_MAX(dm[0], lambda[0] * dm[0]);
    dn[0] = MG_MIN(isqhmin, MG_MAX(isqhmax, dn[0]));
    dn[1] = MG_MAX(dm[1], lambda[0] * dm[1]);
    dn[1] = MG_MIN(isqhmin, MG_MAX(isqhmax, dn[1]));

    /* Intersected metric = P diag(d0,d1){^t}P, P = (vp0, vp1) stored in columns
     */
    mr[0] = dn[0] * vp[0][0] * vp[0][0] + dn[1] * vp[1][0] * vp[1][0];
    mr[1] = dn[0] * vp[0][0] * vp[0][1] + dn[1] * vp[1][0] * vp[1][1];
    mr[2] = dn[0] * vp[0][1] * vp[0][1] + dn[1] * vp[1][1] * vp[1][1];

    return 1;
  }

  /* Second case : both eigenvalues of imn are distinct ; theory says qf
     associated to m and n are diagonalizable in basis (vp0, vp1) - the
     coreduction basis */
  else if (order == 1)
  {
    /* Compute diagonal values in simultaneous reduction basis */
    dm[0] = m[0] * vp[0][0] * vp[0][0] + 2.0 * m[1] * vp[0][0] * vp[0][1] +
            m[2] * vp[0][1] * vp[0][1];
    dm[1] = m[0] * vp[1][0] * vp[1][0] + 2.0 * m[1] * vp[1][0] * vp[1][1] +
            m[2] * vp[1][1] * vp[1][1];
    dn[0] = n[0] * vp[0][0] * vp[0][0] + 2.0 * n[1] * vp[0][0] * vp[0][1] +
            n[2] * vp[0][1] * vp[0][1];
    dn[1] = n[0] * vp[1][0] * vp[1][0] + 2.0 * n[1] * vp[1][0] * vp[1][1] +
            n[2] * vp[1][1] * vp[1][1];

    /* Diagonal values of the intersected metric */
    d0 = MG_MAX(dm[0], dn[0]);
    d0 = MG_MIN(isqhmin, MG_MAX(d0, isqhmax));

    d1 = MG_MAX(dm[1], dn[1]);
    d1 = MG_MIN(isqhmin, MG_MAX(d1, isqhmax));

    /* Intersected metric = tP^-1 diag(d0,d1)P^-1, P = (vp0, vp1) stored in
     * columns */
    det = vp[0][0] * vp[1][1] - vp[0][1] * vp[1][0];
    if (fabs(det) < MMG5_EPS)
      return 0;
    det = 1.0 / det;

    ip[0] = vp[1][1] * det;
    ip[1] = -vp[1][0] * det;
    ip[2] = -vp[0][1] * det;
    ip[3] = vp[0][0] * det;

    mr[0] = d0 * ip[0] * ip[0] + d1 * ip[2] * ip[2];
    mr[1] = d0 * ip[0] * ip[1] + d1 * ip[2] * ip[3];
    mr[2] = d0 * ip[1] * ip[1] + d1 * ip[3] * ip[3];
  }
  return 1;
}

/**
 * \param m pointer to a 3x3 symetric matrix
 * \param mi pointer to the computed 3x3 matrix.
 *
 * Invert \a m (3x3 symetric matrix) and store the result on \a mi
 *
 */
int MMG5_invmat(double *m, double *mi)
{
  double aa, bb, cc, det, vmax, maxx;
  int    k;

  /* check diagonal matrices */
  vmax = fabs(m[1]);
  maxx = fabs(m[2]);
  if (maxx > vmax)
    vmax = maxx;
  maxx = fabs(m[4]);
  if (maxx > vmax)
    vmax = maxx;
  if (vmax < MMG5_EPS)
  {
    mi[0] = 1. / m[0];
    mi[3] = 1. / m[3];
    mi[5] = 1. / m[5];
    mi[1] = mi[2] = mi[4] = 0.0;
    return 1;
  }

  /* check ill-conditionned matrix */
  vmax = fabs(m[0]);
  for (k = 1; k < 6; k++)
  {
    maxx = fabs(m[k]);
    if (maxx > vmax)
      vmax = maxx;
  }
  if (vmax == 0.0)
    return 0;

  /* compute sub-dets */
  aa  = m[3] * m[5] - m[4] * m[4];
  bb  = m[4] * m[2] - m[1] * m[5];
  cc  = m[1] * m[4] - m[2] * m[3];
  det = m[0] * aa + m[1] * bb + m[2] * cc;
  if (fabs(det) < MMG5_EPSD2)
    return 0;
  det = 1.0 / det;

  mi[0] = aa * det;
  mi[1] = bb * det;
  mi[2] = cc * det;
  mi[3] = (m[0] * m[5] - m[2] * m[2]) * det;
  mi[4] = (m[1] * m[2] - m[0] * m[4]) * det;
  mi[5] = (m[0] * m[3] - m[1] * m[1]) * det;

  return 1;
}

/**
 * \fn static int newton3(double p[4],double x[3])
 * \brief Find root(s) of a polynomial of degree 3.
 * \param p polynomial coefficients (b=p[2], c=p[1], d=p[0]).
 * \param x root(s) of polynomial.
 * \return 0 if no roots.
 * \return 1 for 3 roots.
 * \return 2 for 2 roots.
 * \return 3 for 1 root.
 *
 * Find root(s) of a polynomial of degree 3: \f$P(x) = x^3+bx^2+cx+d\f$.
 *
 */
static int newton3(double p[4], double x[3])
{
  double        b, c, d, da, db, dc, epsd;
  double        delta, fx, dfx, dxx;
  double        fdx0, fdx1, dx0, dx1, x1, x2, tmp, epsA, epsB;
  int           it, it2, n;
  static int8_t mmgWarn = 0;

  /* coeffs polynomial, a=1 */
  if (p[3] != 1.)
  {
    if (!mmgWarn)
    {
      fprintf(stderr,
              "\n  ## Warning: %s: bad use of newton3 function, polynomial"
              " must be of type P(x) = x^3+bx^2+cx+d.\n",
              __func__);
      mmgWarn = 1;
    }
    return 0;
  }

  b = p[2];
  c = p[1];
  d = p[0];

  /* 1st derivative of f */
  da = 3.0;
  db = 2.0 * b;

  /* solve 2nd order eqn */
  delta = db * db - 4.0 * da * c;
  epsd  = db * db * MG_EIGENV_EPS10;

  /* inflexion (f'(x)=0, x=-b/2a) */
  x1 = -db / 6.0f;

  n = 1;
  if (delta > epsd)
  {
    delta = sqrt(delta);
    dx0   = (-db + delta) / 6.0;
    dx1   = (-db - delta) / 6.0;
    /* Horner */
    fdx0 = d + dx0 * (c + dx0 * (b + dx0));
    fdx1 = d + dx1 * (c + dx1 * (b + dx1));


    x[2] = -b - 2.0 * dx0;
    tmp  = -b - 2.0 * dx1;
    if (fabs(fdx0) < MG_EIGENV_EPS27 ||
        (fabs(fdx0) < MG_EIGENV_EPS13 && (dx0 > 0. && x[2] > 0.)))
    {
      /* dx0: double root, compute single root */
      n    = 2;
      x[0] = dx0;
      x[1] = dx0;
      /* check if P(x) = 0 */
      fx = d + x[2] * (c + x[2] * (b + x[2]));
      if (fabs(fx) > MG_EIGENV_EPS10)
      {
#ifdef DEBUG
        fprintf(stderr,
                "\n  ## Error: %s: ERR 9100, newton3: fx= %E.\n",
                __func__,
                fx);
#endif
        return 0;
      }
      return n;
    }
    else if (fabs(fdx1) < MG_EIGENV_EPS27 ||
             (fabs(fdx1) < MG_EIGENV_EPS13 && (dx1 > 0. && tmp > 0.)))
    {
      /* dx1: double root, compute single root */
      n    = 2;
      x[0] = dx1;
      x[1] = dx1;
      x[2] = tmp;
      /* check if P(x) = 0 */
      fx = d + x[2] * (c + x[2] * (b + x[2]));
      if (fabs(fx) > MG_EIGENV_EPS10)
      {
#ifdef DEBUG
        fprintf(stderr,
                "\n  ## Error: %s: ERR 9100, newton3: fx= %E.\n",
                __func__,
                fx);
#endif
        return 0;
      }
      return n;
    }
  }

  else if (fabs(delta) < db * db * 1.e-20 || (fabs(delta) < epsd && x1 > 0.))
  {
    /* triple root */
    n    = 3;
    x[0] = x1;
    x[1] = x1;
    x[2] = x1;
    /* check if P(x) = 0 */
    fx = d + x[0] * (c + x[0] * (b + x[0]));
    if (fabs(fx) > MG_EIGENV_EPS10)
    {
#ifdef DEBUG
      fprintf(stderr,
              "\n  ## Error: %s: ERR 9100, newton3: fx= %E.\n",
              __func__,
              fx);
#endif
      return 0;
    }
    return n;
  }

  else
  {
#ifdef DEBUG
    fprintf(stderr,
            "\n  ## Error: %s: ERR 9101, newton3: no real roots.\n",
            __func__);
#endif
    return 0;
  }

  /* Newton method: find one root (middle)
     starting point: P"(x)=0 */
  x1  = -b / 3.0;
  dfx = c + b * x1;
  fx  = d + x1 * (c - 2.0 * x1 * x1);

  epsA = MG_EIGENV_EPS13;
  epsB = MG_EIGENV_EPS10;

  it2 = 0;
newton:

  it = 0;
  do
  {
    x2 = x1 - fx / dfx;
    fx = d + x2 * (c + x2 * (b + x2));
    if (fabs(fx) < epsA && x2 > 0.)
    {
      x[0] = x2;
      break;
    }
    dfx = c + x2 * (db + da * x2);

    /* check for break-off condition */
    dxx = fabs((x2 - x1) / x2);
    if (dxx < epsB && x2 > 0.)
    {
      x[0] = x2;

      /* Check accuracy for 1e-5 precision only (we don't want to fail for
       * smaller precision) */
      if (fabs(fx) > MG_EIGENV_EPS10)
      {
        fprintf(stderr,
                "\n  ## Error: %s: ERR 9102, newton3, no root found"
                " (fx %E).\n",
                __func__,
                fx);
        return 0;
      }
      break;
    }
    else
      x1 = x2;
  }
  while (++it < MAXTOU);

  if (it == MAXTOU)
  {
    x[0] = x1;
    fx   = d + x1 * (c + (x1 * (b + x1)));
    /* Check accuracy for 1e-5 precision only (we don't want to fail for smaller
     * precision) */
    if (fabs(fx) > MG_EIGENV_EPS10)
    {
      fprintf(stderr,
              "\n  ## Error: %s: ERR 9102, newton3, no root found"
              " (fx %E).\n",
              __func__,
              fx);
      return 0;
    }
  }

  /* solve 2nd order equation
     P(x) = (x-sol(1))* (x^2+bb*x+cc)  */
  db    = b + x[0];
  dc    = c + x[0] * db;
  delta = db * db - 4.0 * dc;

  if (delta <= 0.0)
  {
    fprintf(stderr,
            "\n  ## Error: %s: ERR 9103, newton3, det = 0.\n",
            __func__);
    return 0;
  }

  delta = sqrt(delta);
  x[1]  = 0.5 * (-db + delta);
  x[2]  = 0.5 * (-db - delta);

  while (++it2 < 5 && ((x[0] <= 0 && fabs(x[0]) <= MG_EIGENV_EPS5) ||
                       (x[1] <= 0 && fabs(x[1]) <= MG_EIGENV_EPS5) ||
                       (x[2] <= 0 && fabs(x[2]) <= MG_EIGENV_EPS5)))
  {
    /* Get back to the newton method with increased accuracy */
    epsA /= 10;
    epsB /= 10;
    goto newton;
  }

#ifdef DEBUG
  /* check for root accuracy */
  fx = d + x[1] * (c + x[1] * (b + x[1]));
  if (fabs(fx) > MG_EIGENV_EPS10)
  {
    fprintf(stderr,
            "\n  ## Error: %s: ERR 9104, newton3: fx= %E  x= %E.\n",
            __func__,
            fx,
            x[1]);
    return 0;
  }
  fx = d + x[2] * (c + x[2] * (b + x[2]));
  if (fabs(fx) > MG_EIGENV_EPS10)
  {
    fprintf(stderr,
            "\n  ## Error: %s: ERR 9104, newton3: fx= %E  x= %E.\n",
            __func__,
            fx,
            x[2]);
    return 0;
  }
#endif

  return n;
}

/**
 * \param m symmetric matrix
 * \param n symmetric matrix
 * \param mn result
 *
 * Compute product m*n (mn stored by rows for consistency with MMG5_eigenv3d).
 *
 */
void MMG5_mn(double m[6], double n[6], double mn[9])
{
  mn[0] = m[0] * n[0] + m[1] * n[1] + m[2] * n[2];
  mn[1] = m[0] * n[1] + m[1] * n[3] + m[2] * n[4];
  mn[2] = m[0] * n[2] + m[1] * n[4] + m[2] * n[5];
  mn[3] = m[1] * n[0] + m[3] * n[1] + m[4] * n[2];
  mn[4] = m[1] * n[1] + m[3] * n[3] + m[4] * n[4];
  mn[5] = m[1] * n[2] + m[3] * n[4] + m[4] * n[5];
  mn[6] = m[2] * n[0] + m[4] * n[1] + m[5] * n[2];
  mn[7] = m[2] * n[1] + m[4] * n[3] + m[5] * n[4];
  mn[8] = m[2] * n[2] + m[4] * n[4] + m[5] * n[5];

  return;
}

/**
 * \param r 3x3 matrix
 * \param m symetric matrix
 * \param mr result
 *
 * \return 1
 *
 * Compute product R*M*tR when M is symmetric
 *
 */
inline int MMG5_rmtr(double r[3][3], double m[6], double mr[6])
{
  double n[3][3];

  n[0][0] = m[0] * r[0][0] + m[1] * r[0][1] + m[2] * r[0][2];
  n[1][0] = m[1] * r[0][0] + m[3] * r[0][1] + m[4] * r[0][2];
  n[2][0] = m[2] * r[0][0] + m[4] * r[0][1] + m[5] * r[0][2];

  n[0][1] = m[0] * r[1][0] + m[1] * r[1][1] + m[2] * r[1][2];
  n[1][1] = m[1] * r[1][0] + m[3] * r[1][1] + m[4] * r[1][2];
  n[2][1] = m[2] * r[1][0] + m[4] * r[1][1] + m[5] * r[1][2];

  n[0][2] = m[0] * r[2][0] + m[1] * r[2][1] + m[2] * r[2][2];
  n[1][2] = m[1] * r[2][0] + m[3] * r[2][1] + m[4] * r[2][2];
  n[2][2] = m[2] * r[2][0] + m[4] * r[2][1] + m[5] * r[2][2];

  mr[0] = r[0][0] * n[0][0] + r[0][1] * n[1][0] + r[0][2] * n[2][0];
  mr[1] = r[0][0] * n[0][1] + r[0][1] * n[1][1] + r[0][2] * n[2][1];
  mr[2] = r[0][0] * n[0][2] + r[0][1] * n[1][2] + r[0][2] * n[2][2];
  mr[3] = r[1][0] * n[0][1] + r[1][1] * n[1][1] + r[1][2] * n[2][1];
  mr[4] = r[1][0] * n[0][2] + r[1][1] * n[1][2] + r[1][2] * n[2][2];
  mr[5] = r[2][0] * n[0][2] + r[2][1] * n[1][2] + r[2][2] * n[2][2];

  return 1;
}

/**
 * \param dim size of the array
 * \param a first array
 * \param b second array
 * \param result scalar product of the two arrays
 *
 * Compute scalar product of two double precision arrays.
 */
void MMG5_dotprod(int8_t dim, double *a, double *b, double *result)
{
  *result = 0.0;
  for (int8_t i = 0; i < dim; i++)
    *result += a[i] * b[i];
}

/**
 * \param a first array
 * \param b second array
 * \param result cross product of the two arrays
 *
 * Compute cross product of two double precision arrays in 3D.
 */
void MMG5_crossprod3d(double *a, double *b, double *result)
{
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * \param mat pointer to a 3x3 matrix.
 * \param lambda eigenvalues.
 * \param v eigenvectors.
 * \param w1 temporary array to perform the matrix cross product.
 * \param w2 temporary array to perform the matrix cross product.
 * \param w3 temporary array to perform the matrix cross product.
 * \param maxm maximal value of the matrix used for normalization.
 * \param order order of eigenvalues (1,2,3) or 0 if failed.
 * \param symmat 0 if matrix is not symetric, 1 otherwise.
 *
 * \return 1 if success, 0 if fail.
 *
 * Check the accuracy of the eigenvalues and vectors computation of a 3x3 matrix
 * (symetric).
 *
 */
static int MMG5_check_accuracy(double mat[6],
                               double lambda[3],
                               double v[3][3],
                               double w1[3],
                               double w2[3],
                               double w3[3],
                               double maxm,
                               int    order,
                               int    symmat)
{
  double err, tmpx, tmpy, tmpz;
  double m[6];
  int    i, j, k;

  if (!symmat)
    return 1;

  k = 0;
  for (i = 0; i < 3; i++)
  {
    for (j = i; j < 3; j++)
    {
      m[k++] = lambda[0] * v[0][i] * v[0][j] + lambda[1] * v[1][i] * v[1][j] +
               lambda[2] * v[2][i] * v[2][j];
    }
  }
  err = fabs(mat[0] - m[0]);
  for (i = 1; i < 6; i++)
    if (fabs(m[i] - mat[i]) > err)
      err = fabs(m[i] - mat[i]);

  if (err > 1.e03 * maxm)
  {
    fprintf(stderr,
            "\n  ## Error: %s:\nProbleme eigenv3: err= %f\n",
            __func__,
            err * maxm);
    fprintf(stderr, "\n  ## Error: %s:mat depart :\n", __func__);
    fprintf(stderr,
            "\n  ## Error: %s:%13.6f  %13.6f  %13.6f\n",
            __func__,
            mat[0],
            mat[1],
            mat[2]);
    fprintf(stderr,
            "\n  ## Error: %s:%13.6f  %13.6f  %13.6f\n",
            __func__,
            mat[1],
            mat[3],
            mat[4]);
    fprintf(stderr,
            "\n  ## Error: %s:%13.6f  %13.6f  %13.6f\n",
            __func__,
            mat[2],
            mat[4],
            mat[5]);
    fprintf(stderr, "\n  ## Error: %s:mat finale :\n", __func__);
    fprintf(stderr,
            "\n  ## Error: %s:%13.6f  %13.6f  %13.6f\n",
            __func__,
            m[0],
            m[1],
            m[2]);
    fprintf(stderr,
            "\n  ## Error: %s:%13.6f  %13.6f  %13.6f\n",
            __func__,
            m[1],
            m[3],
            m[4]);
    fprintf(stderr,
            "\n  ## Error: %s:%13.6f  %13.6f  %13.6f\n",
            __func__,
            m[2],
            m[4],
            m[5]);
    fprintf(stderr,
            "\n  ## Error: %s:lambda : %f %f %f\n",
            __func__,
            lambda[0],
            lambda[1],
            lambda[2]);
    fprintf(stderr, "\n  ## Error: %s: ordre %d\n", __func__, order);
    fprintf(stderr, "\n  ## Error: %s:\nOrtho:\n", __func__);
    fprintf(stderr,
            "\n  ## Error: %s:v1.v2 = %.14f\n",
            __func__,
            v[0][0] * v[1][0] + v[0][1] * v[1][1] + v[0][2] * v[1][2]);
    fprintf(stderr,
            "\n  ## Error: %s:v1.v3 = %.14f\n",
            __func__,
            v[0][0] * v[2][0] + v[0][1] * v[2][1] + v[0][2] * v[2][2]);
    fprintf(stderr,
            "\n  ## Error: %s:v2.v3 = %.14f\n",
            __func__,
            v[1][0] * v[2][0] + v[1][1] * v[2][1] + v[1][2] * v[2][2]);

    fprintf(stderr, "\n  ## Error: %s:Consistency\n", __func__);
    for (i = 0; i < 3; i++)
    {
      tmpx =
        v[0][i] * m[0] + v[1][i] * m[1] + v[2][i] * m[2] - lambda[i] * v[0][i];
      tmpy =
        v[0][i] * m[1] + v[1][i] * m[3] + v[2][i] * m[4] - lambda[i] * v[1][i];
      tmpz =
        v[0][i] * m[2] + v[1][i] * m[4] + v[2][i] * m[5] - lambda[i] * v[2][i];
      fprintf(stderr,
              "\n  ## Error: %s: Av %d - lambda %d *v %d = %f %f %f\n",
              __func__,
              i,
              i,
              i,
              tmpx,
              tmpy,
              tmpz);

      fprintf(stderr,
              "\n  ## Error: %s:w1 %f %f %f\n",
              __func__,
              w1[0],
              w1[1],
              w1[2]);
      fprintf(stderr,
              "\n  ## Error: %s:w2 %f %f %f\n",
              __func__,
              w2[0],
              w2[1],
              w2[2]);
      fprintf(stderr,
              "\n  ## Error: %s:w3 %f %f %f\n",
              __func__,
              w3[0],
              w3[1],
              w3[2]);
    }
    return 0;
  }

  return 1;
}

/**
 * \brief Find eigenvalues and vectors of a 3x3 matrix.
 * \param symmat 0 if matrix is not symetric, 1 otherwise.
 * \param mat pointer to the matrix.
 * \param lambda eigenvalues.
 * \param v eigenvectors.
 *
 * \return order of eigenvalues (1,2,3) or 0 if failed.
 *
 * \remark the i^{th} eigenvector is stored in v[i][.].
 *
 */
int MMG5_eigenv3d(int symmat, double *mat, double lambda[3], double v[3][3])
{
  double a11, a12, a13, a21, a22, a23, a31, a32, a33;
  double aa, bb, cc, dd, ee, ii, vx1[3], vx2[3], vx3[3], dd1, dd2, dd3;
  double maxd, maxm, valm, p[4], w1[3], w2[3], w3[3], epsd;
  int    k, n;

  epsd = MG_EIGENV_EPS13;

  w1[0] = w1[1] = w1[2] = 0;
  w2[0] = w2[1] = w2[2] = 0;
  w3[0] = w3[1] = w3[2] = 0;
  /* default */
  memcpy(v, Id, 9 * sizeof(double));
  if (symmat)
  {
    lambda[0] = (double)mat[0];
    lambda[1] = (double)mat[3];
    lambda[2] = (double)mat[5];

    maxm = fabs(mat[0]);
    for (k = 1; k < 6; k++)
    {
      valm = fabs(mat[k]);
      if (valm > maxm)
        maxm = valm;
    }
    /* single float accuracy if sufficient, else double float accuracy */
    if (maxm < MG_EIGENV_EPS5e6)
    {
      if (lambda[0] > 0. && lambda[1] > 0. && lambda[2] > 0.)
      {
        return 1;
      }
      else if (maxm < MG_EIGENV_EPS13)
      {
        return 0;
      }
      epsd = MG_EIGENV_EPS27;
    }

    /* normalize matrix */
    dd  = 1.0 / maxm;
    a11 = mat[0] * dd;
    a12 = mat[1] * dd;
    a13 = mat[2] * dd;
    a22 = mat[3] * dd;
    a23 = mat[4] * dd;
    a33 = mat[5] * dd;

    /* diagonal matrix */
    maxd = fabs(a12);
    valm = fabs(a13);
    if (valm > maxd)
      maxd = valm;
    valm = fabs(a23);
    if (valm > maxd)
      maxd = valm;
    if (maxd < epsd)
      return 1;

    a21 = a12;
    a31 = a13;
    a32 = a23;

    /* build characteristic polynomial
       P(X) = X^3 - trace X^2 + (somme des mineurs)X - det = 0 */
    aa   = a11 * a22;
    bb   = a23 * a32;
    cc   = a12 * a21;
    dd   = a13 * a31;
    p[0] = a11 * bb + a33 * (cc - aa) + a22 * dd - 2.0 * a12 * a13 * a23;
    p[1] = a11 * (a22 + a33) + a22 * a33 - bb - cc - dd;
    p[2] = -a11 - a22 - a33;
    p[3] = 1.0;
  }
  else
  {
    lambda[0] = (double)mat[0];
    lambda[1] = (double)mat[4];
    lambda[2] = (double)mat[8];

    maxm = fabs(mat[0]);
    for (k = 1; k < 9; k++)
    {
      valm = fabs(mat[k]);
      if (valm > maxm)
        maxm = valm;
    }

    /* single float accuracy if sufficient, else double float accuracy */
    if (maxm < MG_EIGENV_EPS5e6)
    {
      if (lambda[0] > 0. && lambda[1] > 0. && lambda[2] > 0.)
      {
        return 1;
      }
      else if (maxm < MG_EIGENV_EPS13)
      {
        return 0;
      }
      epsd = MG_EIGENV_EPS27;
    }

    /* normalize matrix */
    dd  = 1.0 / maxm;
    a11 = mat[0] * dd;
    a12 = mat[1] * dd;
    a13 = mat[2] * dd;
    a21 = mat[3] * dd;
    a22 = mat[4] * dd;
    a23 = mat[5] * dd;
    a31 = mat[6] * dd;
    a32 = mat[7] * dd;
    a33 = mat[8] * dd;

    /* diagonal matrix */
    maxd = fabs(a12);
    valm = fabs(a13);
    if (valm > maxd)
      maxd = valm;
    valm = fabs(a23);
    if (valm > maxd)
      maxd = valm;
    valm = fabs(a21);
    if (valm > maxd)
      maxd = valm;
    valm = fabs(a31);
    if (valm > maxd)
      maxd = valm;
    valm = fabs(a32);
    if (valm > maxd)
      maxd = valm;
    if (maxd < epsd)
      return 1;

    /* build characteristic polynomial
       P(X) = X^3 - trace X^2 + (somme des mineurs)X - det = 0 */
    aa = a22 * a33 - a23 * a32;
    bb = a23 * a31 - a21 * a33;
    cc = a21 * a32 - a31 * a22;
    ee = a11 * a33 - a13 * a31;
    ii = a11 * a22 - a12 * a21;

    p[0] = -a11 * aa - a12 * bb - a13 * cc;
    p[1] = aa + ee + ii;
    p[2] = -a11 - a22 - a33;
    p[3] = 1.0;
  }

  /* solve polynomial (find roots using newton) */
  n = newton3(p, lambda);
  if (n <= 0)
    return 0;

  /* compute eigenvectors:
     an eigenvalue belong to orthogonal of Im(A-lambda*Id) */
  v[0][0] = 1.0;
  v[0][1] = v[0][2] = 0.0;
  v[1][1]           = 1.0;
  v[1][0] = v[1][2] = 0.0;
  v[2][2]           = 1.0;
  v[2][0] = v[2][1] = 0.0;

  w1[1] = a12;
  w1[2] = a13;
  w2[0] = a21;
  w2[2] = a23;
  w3[0] = a31;
  w3[1] = a32;

  if (n == 1)
  {
    /* vk = crsprd(wi,wj) */
    for (k = 0; k < 3; k++)
    {
      w1[0] = a11 - lambda[k];
      w2[1] = a22 - lambda[k];
      w3[2] = a33 - lambda[k];

      /* cross product vectors in (Im(A-lambda(i) Id) ortho */
      MMG5_crossprod3d(w1, w3, vx1);
      MMG5_dotprod(3, vx1, vx1, &dd1);

      MMG5_crossprod3d(w1, w2, vx2);
      MMG5_dotprod(3, vx2, vx2, &dd2);

      MMG5_crossprod3d(w2, w3, vx3);
      MMG5_dotprod(3, vx3, vx3, &dd3);

      /* find vector of max norm */
      if (dd1 > dd2)
      {
        if (dd1 > dd3)
        {
          dd1     = 1.0 / sqrt(dd1);
          v[k][0] = vx1[0] * dd1;
          v[k][1] = vx1[1] * dd1;
          v[k][2] = vx1[2] * dd1;
        }
        else
        {
          dd3     = 1.0 / sqrt(dd3);
          v[k][0] = vx3[0] * dd3;
          v[k][1] = vx3[1] * dd3;
          v[k][2] = vx3[2] * dd3;
        }
      }
      else
      {
        if (dd2 > dd3)
        {
          dd2     = 1.0 / sqrt(dd2);
          v[k][0] = vx2[0] * dd2;
          v[k][1] = vx2[1] * dd2;
          v[k][2] = vx2[2] * dd2;
        }
        else
        {
          dd3     = 1.0 / sqrt(dd3);
          v[k][0] = vx3[0] * dd3;
          v[k][1] = vx3[1] * dd3;
          v[k][2] = vx3[2] * dd3;
        }
      }
    }
  }

  /* (vp1,vp2) double,  vp3 simple root */
  else if (n == 2)
  {
    /* basis vectors of Im(tA-lambda[2]*I) */
    double z1[3], z2[3];

    /** rows of A-lambda[2]*I */
    w1[0] = a11 - lambda[2];
    w2[1] = a22 - lambda[2];
    w3[2] = a33 - lambda[2];

    /* ker(A-lambda[2]*I) has dimension 1 and it is orthogonal to
     * Im(tA-lambda[2]*I), which has dimension 2.
     * So the eigenvector vp[2] can be computed as the cross product of the two
     * linearly independent rows of (A-lambda[2]*I).
     *
     * Compute all pairwise cross products of the rows of (A-lambda[2]*I), and
     * pick the one with maximum norm (the other two will have zero norm, but
     * this is tricky to detect numerically due to cancellation errors). */
    MMG5_crossprod3d(w1, w3, vx1);
    MMG5_dotprod(3, vx1, vx1, &dd1);

    MMG5_crossprod3d(w1, w2, vx2);
    MMG5_dotprod(3, vx2, vx2, &dd2);

    MMG5_crossprod3d(w2, w3, vx3);
    MMG5_dotprod(3, vx3, vx3, &dd3);

    /* find vector of max norm to pick the two linearly independent rows */
    if (dd1 > dd2)
    {
      if (dd1 > dd3)
      {
        dd1     = 1.0 / sqrt(dd1);
        v[2][0] = vx1[0] * dd1;
        v[2][1] = vx1[1] * dd1;
        v[2][2] = vx1[2] * dd1;
        memcpy(z1, w1, 3 * sizeof(double));
        memcpy(z2, w3, 3 * sizeof(double));
      }
      else
      {
        dd3     = 1.0 / sqrt(dd3);
        v[2][0] = vx3[0] * dd3;
        v[2][1] = vx3[1] * dd3;
        v[2][2] = vx3[2] * dd3;
        memcpy(z1, w2, 3 * sizeof(double));
        memcpy(z2, w3, 3 * sizeof(double));
      }
    }
    else
    {
      if (dd2 > dd3)
      {
        dd2     = 1.0 / sqrt(dd2);
        v[2][0] = vx2[0] * dd2;
        v[2][1] = vx2[1] * dd2;
        v[2][2] = vx2[2] * dd2;
        memcpy(z1, w1, 3 * sizeof(double));
        memcpy(z2, w2, 3 * sizeof(double));
      }
      else
      {
        dd3     = 1.0 / sqrt(dd3);
        v[2][0] = vx3[0] * dd3;
        v[2][1] = vx3[1] * dd3;
        v[2][2] = vx3[2] * dd3;
        memcpy(z1, w2, 3 * sizeof(double));
        memcpy(z2, w3, 3 * sizeof(double));
      }
    }
    /* The two linearly independent rows provide a basis for Im(tA-lambda[2]*I).
     * Normalize them to reduce roundoff errors. */
    MMG5_dotprod(3, z1, z1, &dd1);
    dd1 = 1.0 / sqrt(dd1);
    z1[0] *= dd1;
    z1[1] *= dd1;
    z1[2] *= dd1;
    MMG5_dotprod(3, z2, z2, &dd2);
    dd2 = 1.0 / sqrt(dd2);
    z2[0] *= dd2;
    z2[1] *= dd2;
    z2[2] *= dd2;


    /** rows of A-lambda[0]*I */
    w1[0] = a11 - lambda[0];
    w2[1] = a22 - lambda[0];
    w3[2] = a33 - lambda[0];

    /* ker(A-lambda[0]*I) has dimension 2 and it is orthogonal to
     * Im(tA-lambda[0]*I), which has dimension 1.
     * Eigenvectors vp[0],vp[1] belong to ker(A-lambda[0]*I) and can't belong to
     * ker(A-lambda[2]*I) since eigenvalue lambda[2] is distinct. Thus, by
     * orthogonality, the vectors belonging to Im(tA-lambda[2]*I) can't belong
     * to Im(tA-lambda[0]*I).
     * Denoting as c20 and c21 the two basis vectors for Im(tA-lambda[2]*I), and
     * as c0 the only basis vector for Im(tA-lambda[0]*I), two _distinct_
     * eigenvectors vp[0] and vp[0] in ker(A-lambda[0]*I) can thus be computed
     * as:
     *   vp[0] = c0 x c20
     *   vp[1] = c0 x c21
     * (Stated differently, vp[0] and vp[1] would be colinear only if
     * Im(tA-lambda[0]*I) belonged to Im(tA-lambda[2]), which would imply that
     * ker(A-lambda[2]*I) belong to ker(A-lambda[0]*I), that is not possible).
     *
     * Find the basis of Im(tA-lambda[0]*I) as the row with maximum projection
     * on vp[2] (this helps in selecting a row that does not belong to
     * Im(tA-lambda[2]*I) in case lambda[2] is numerically not well separated
     * from lambda[0]=lambda[1]).
     */
    MMG5_dotprod(3, w1, v[2], &dd1);
    MMG5_dotprod(3, w2, v[2], &dd2);
    MMG5_dotprod(3, w3, v[2], &dd3);
    dd1 = fabs(dd1);
    dd2 = fabs(dd2);
    dd3 = fabs(dd3);

    /* find vector with max projection to pick the linearly independent row */
    if (dd1 > dd2)
    {
      if (dd1 > dd3)
      {
        MMG5_dotprod(3, w1, w1, &dd1);
        dd1    = 1.0 / sqrt(dd1);
        vx1[0] = w1[0] * dd1;
        vx1[1] = w1[1] * dd1;
        vx1[2] = w1[2] * dd1;
      }
      else
      {
        MMG5_dotprod(3, w3, w3, &dd3);
        dd3    = 1.0 / sqrt(dd3);
        vx1[0] = w3[0] * dd3;
        vx1[1] = w3[1] * dd3;
        vx1[2] = w3[2] * dd3;
      }
    }
    else
    {
      if (dd2 > dd3)
      {
        MMG5_dotprod(3, w2, w2, &dd2);
        dd2    = 1.0 / sqrt(dd2);
        vx1[0] = w2[0] * dd2;
        vx1[1] = w2[1] * dd2;
        vx1[2] = w2[2] * dd2;
      }
      else
      {
        MMG5_dotprod(3, w3, w3, &dd3);
        dd3    = 1.0 / sqrt(dd3);
        vx1[0] = w3[0] * dd3;
        vx1[1] = w3[1] * dd3;
        vx1[2] = w3[2] * dd3;
      }
    }
    /* cross product of the first basis vector of Im(tA-lambda[2]*I) with the
     * basis vector of Im(tA-lambda[0]) */
    MMG5_crossprod3d(z1, vx1, v[0]);
    MMG5_dotprod(3, v[0], v[0], &dd1);
    assert(dd1 > MG_EIGENV_EPS27);
    dd1 = 1.0 / sqrt(dd1);
    v[0][0] *= dd1;
    v[0][1] *= dd1;
    v[0][2] *= dd1;

    /* 3rd vector as the cross product of the second basis vector of
     * Im(tA-lambda[2]*I) with the basis vector of Im(tA-lambda[0]) */
    MMG5_crossprod3d(vx1, z2, v[1]);
    MMG5_dotprod(3, v[1], v[1], &dd2);
    assert(dd2 > MG_EIGENV_EPS27);
    dd2 = 1.0 / sqrt(dd2);
    v[1][0] *= dd2;
    v[1][1] *= dd2;
    v[1][2] *= dd2;

    /* enforce orthogonality in the symmetric case (can't prove that c20 and
     * c21 are orthogonal in a general symmetric case), the result will still
     * belong to ker(A-lambda[0]*I) */
    if (symmat)
    {
      MMG5_dotprod(3, v[1], v[0], &dd1);
      v[1][0] -= dd1 * v[0][0];
      v[1][1] -= dd1 * v[0][1];
      v[1][2] -= dd1 * v[0][2];
      /* normalize again */
      MMG5_dotprod(3, v[1], v[1], &dd2);
      assert(dd2 > MG_EIGENV_EPS27);
      dd2 = 1.0 / sqrt(dd2);
      v[1][0] *= dd2;
      v[1][1] *= dd2;
      v[1][2] *= dd2;
    }
  }

  lambda[0] *= maxm;
  lambda[1] *= maxm;
  lambda[2] *= maxm;

  /* check accuracy */
  if (getenv("MMG_EIGENV_DDEBUG") && symmat)
  {
    if (!MMG5_check_accuracy(mat, lambda, v, w1, w2, w3, maxm, n, symmat))
      return 0;
  }

  return n;
}

/**
 * \param mesh pointer to the mesh (REMOVED BECAUSE UNUSED)
 * \param m first matrix
 * \param n second matrix
 * \param dm eigenvalues of m in the coreduction basis (to fill)
 * \param dn eigenvalues of n in the coreduction basis (to fill)
 * \param vp coreduction basis (to fill)
 *
 * \return 0 if fail 1 otherwise.
 *
 * Perform simultaneous reduction of matrices \a m and \a n.
 *
 */
int MMG5_simred3d(double *m,
                  double *n,
                  double  dm[3],
                  double  dn[3],
                  double  vp[3][3])
{
  double        lambda[3], im[6], imn[9];
  int           order;
  static int8_t mmgWarn0 = 0;

  /* Compute imn = M^{-1}N */
  if (!MMG5_invmat(m, im))
  {
    if (!mmgWarn0)
    {
      mmgWarn0 = 1;
      fprintf(stderr,
              "\n  ## Warning: %s: unable to invert the matrix.\n",
              __func__);
    }
    return 0;
  }

  MMG5_mn(im, n, imn);

  /* Find eigenvalues of imn */
  order = MMG5_eigenv3d(0, imn, lambda, vp);

  if (!order)
  {
    if (!mmgWarn0)
    {
      mmgWarn0 = 1;
      fprintf(stderr,
              "\n  ## Warning: %s: at least 1 failing"
              " simultaneous reduction.\n",
              __func__);
    }
    return 0;
  }

  if (order == 3)
  {
    /* First case : matrices m and n are homothetic: n = lambda0*m */
    if ((fabs(m[1]) < MMG5_EPS && fabs(m[2]) < MMG5_EPS &&
         fabs(m[4]) < MMG5_EPS))
    {
      /* Subcase where m is diaonal */
      dm[0]    = m[0];
      dm[1]    = m[3];
      dm[2]    = m[5];
      vp[0][0] = 1;
      vp[0][1] = 0;
      vp[0][2] = 0;
      vp[1][0] = 0;
      vp[1][1] = 1;
      vp[1][2] = 0;
      vp[2][0] = 0;
      vp[2][1] = 0;
      vp[2][2] = 1;
    }
    else
    {
      /* Subcase where m is not diagonal; dd,trimn,... are reused */
      MMG5_eigenv3d(1, m, dm, vp);
    }
    /* Eigenvalues of metric n */
    dn[0] = lambda[0] * dm[0];
    dn[1] = lambda[0] * dm[1];
    dn[2] = lambda[0] * dm[2];
  }
  else if (order == 2)
  {
    /* Second case: two eigenvalues of imn are coincident (first two entries of
     * the lambda array) and one is distinct (last entry).
     * Simultaneous reduction gives a block diagonalization. The 2x2 blocks are
     * homothetic and can be diagonalized through the eigenvectors of one of the
     * two blocks. */
    double mred[6], nred[6];
    /* project matrices on the coreduction basis: they should have the
     * block-diagonal form [ m0, m1, 0, m3, 0, m5 ] */
    MMG5_rmtr(vp, m, mred);
    MMG5_rmtr(vp, n, nred);
    /* compute projections on the last eigenvector (that with multiplicity 1) */
    dm[2] = mred[5];
    dn[2] = nred[5];
    /* re-arrange matrices so that the first three entries describe the
     * 2x2 blocks to be diagonalized (the two blocks are homothetic) */
    mred[2] = mred[3];
    nred[2] = nred[3];
    /* diagonalization of the first 2x2 block */
    if (fabs(mred[1]) < MMG5_EPS)
    {
      /* first case: the blocks are diagonal, basis vp is unchanged */
      dm[0] = mred[0];
      dm[1] = mred[2];
    }
    else
    {
      /* second case: the blocks are not diagonal */
      double wp[2][2], up[2][3];
      int8_t i, j, k;
      MMG5_eigensym(mred, dm, wp);
      /* change the basis vp (vp[2] is unchanged) */
      for (j = 0; j < 2; j++)
      {
        for (i = 0; i < 3; i++)
        {
          up[j][i] = 0.;
          for (k = 0; k < 2; k++)
          {
            up[j][i] += vp[k][i] * wp[j][k];
          }
        }
      }
      for (j = 0; j < 2; j++)
      {
        for (i = 0; i < 3; i++)
        {
          vp[j][i] = up[j][i];
        }
      }
    }
    /* homothetic diagonalization of the second 2x2 block */
    dn[0] = lambda[0] * dm[0];
    dn[1] = lambda[0] * dm[1];
  }
  else
  {
    /* Third case: eigenvalues of imn are distinct ; theory says qf associated
       to m and n are diagonalizable in basis (vp[0], vp[1], vp[2]) - the
       coreduction basis */
    /* Compute diagonal values in simultaneous reduction basis */
    dm[0] = m[0] * vp[0][0] * vp[0][0] + 2.0 * m[1] * vp[0][0] * vp[0][1] +
            2.0 * m[2] * vp[0][0] * vp[0][2] + m[3] * vp[0][1] * vp[0][1] +
            2.0 * m[4] * vp[0][1] * vp[0][2] + m[5] * vp[0][2] * vp[0][2];
    dm[1] = m[0] * vp[1][0] * vp[1][0] + 2.0 * m[1] * vp[1][0] * vp[1][1] +
            2.0 * m[2] * vp[1][0] * vp[1][2] + m[3] * vp[1][1] * vp[1][1] +
            2.0 * m[4] * vp[1][1] * vp[1][2] + m[5] * vp[1][2] * vp[1][2];
    dm[2] = m[0] * vp[2][0] * vp[2][0] + 2.0 * m[1] * vp[2][0] * vp[2][1] +
            2.0 * m[2] * vp[2][0] * vp[2][2] + m[3] * vp[2][1] * vp[2][1] +
            2.0 * m[4] * vp[2][1] * vp[2][2] + m[5] * vp[2][2] * vp[2][2];

    dn[0] = n[0] * vp[0][0] * vp[0][0] + 2.0 * n[1] * vp[0][0] * vp[0][1] +
            2.0 * n[2] * vp[0][0] * vp[0][2] + n[3] * vp[0][1] * vp[0][1] +
            2.0 * n[4] * vp[0][1] * vp[0][2] + n[5] * vp[0][2] * vp[0][2];
    dn[1] = n[0] * vp[1][0] * vp[1][0] + 2.0 * n[1] * vp[1][0] * vp[1][1] +
            2.0 * n[2] * vp[1][0] * vp[1][2] + n[3] * vp[1][1] * vp[1][1] +
            2.0 * n[4] * vp[1][1] * vp[1][2] + n[5] * vp[1][2] * vp[1][2];
    dn[2] = n[0] * vp[2][0] * vp[2][0] + 2.0 * n[1] * vp[2][0] * vp[2][1] +
            2.0 * n[2] * vp[2][0] * vp[2][2] + n[3] * vp[2][1] * vp[2][1] +
            2.0 * n[4] * vp[2][1] * vp[2][2] + n[5] * vp[2][2] * vp[2][2];
  }

  assert(dm[0] >= MMG5_EPSD2 && dm[1] >= MMG5_EPSD2 && dm[2] >= MMG5_EPSD2 &&
         "positive eigenvalue");
  assert(dn[0] >= MMG5_EPSD2 && dn[1] >= MMG5_EPSD2 && dn[2] >= MMG5_EPSD2 &&
         "positive eigenvalue");

  if (dm[0] < MMG5_EPSOK || dn[0] < MMG5_EPSOK)
  {
    return 0;
  }
  if (dm[1] < MMG5_EPSOK || dn[1] < MMG5_EPSOK)
  {
    return 0;
  }
  if (dm[2] < MMG5_EPSOK || dn[2] < MMG5_EPSOK)
  {
    return 0;
  }

  return 1;
}

/**
 * \param m initial matrix.
 * \param mi inverted matrix.
 *
 * Invert 3x3 non-symmetric matrix stored in 2 dimensions
 *
 */
int MMG5_invmat33(double m[3][3], double mi[3][3])
{
  double aa, bb, cc, det, vmax, maxx;
  int    k, l;

  /* check ill-conditionned matrix */
  vmax = fabs(m[0][0]);
  for (k = 0; k < 3; k++)
  {
    for (l = 0; l < 3; l++)
    {
      maxx = fabs(m[k][l]);
      if (maxx > vmax)
        vmax = maxx;
    }
  }
  if (vmax == 0.0)
    return 0;

  /* check diagonal matrices */
  /* lower */
  vmax = fabs(m[1][0]);
  maxx = fabs(m[2][0]);
  if (maxx > vmax)
    vmax = maxx;
  maxx = fabs(m[2][1]);
  if (maxx > vmax)
    vmax = maxx;
  /* upper */
  maxx = fabs(m[0][1]);
  if (maxx > vmax)
    vmax = maxx;
  maxx = fabs(m[0][2]);
  if (maxx > vmax)
    vmax = maxx;
  maxx = fabs(m[1][2]);
  if (maxx > vmax)
    vmax = maxx;

  if (vmax < MMG5_EPS)
  {
    mi[0][0] = 1. / m[0][0];
    mi[1][1] = 1. / m[1][1];
    mi[2][2] = 1. / m[2][2];
    mi[1][0] = mi[0][1] = mi[2][0] = mi[0][2] = mi[1][2] = mi[2][1] = 0.0;
    return 1;
  }

  /* compute sub-dets */
  aa  = m[1][1] * m[2][2] - m[2][1] * m[1][2];
  bb  = m[2][1] * m[0][2] - m[0][1] * m[2][2];
  cc  = m[0][1] * m[1][2] - m[1][1] * m[0][2];
  det = m[0][0] * aa + m[1][0] * bb + m[2][0] * cc;
  if (fabs(det) < MMG5_EPSD)
    return 0;
  det = 1.0 / det;

  mi[0][0] = aa * det;
  mi[0][1] = bb * det;
  mi[0][2] = cc * det;
  mi[1][0] = (m[2][0] * m[1][2] - m[1][0] * m[2][2]) * det;
  mi[1][1] = (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * det;
  mi[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * det;
  mi[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * det;
  mi[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * det;
  mi[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * det;

  /* Check results */
#ifndef NDEBUG
  double res[3][3];

  res[0][0] = m[0][0] * mi[0][0] + m[0][1] * mi[1][0] + m[0][2] * mi[2][0];
  res[0][1] = m[0][0] * mi[0][1] + m[0][1] * mi[1][1] + m[0][2] * mi[2][1];
  res[0][2] = m[0][0] * mi[0][2] + m[0][1] * mi[1][2] + m[0][2] * mi[2][2];

  res[1][0] = m[1][0] * mi[0][0] + m[1][1] * mi[1][0] + m[1][2] * mi[2][0];
  res[1][1] = m[1][0] * mi[0][1] + m[1][1] * mi[1][1] + m[1][2] * mi[2][1];
  res[1][2] = m[1][0] * mi[0][2] + m[1][1] * mi[1][2] + m[1][2] * mi[2][2];

  res[2][0] = m[2][0] * mi[0][0] + m[2][1] * mi[1][0] + m[2][2] * mi[2][0];
  res[2][1] = m[2][0] * mi[0][1] + m[2][1] * mi[1][1] + m[2][2] * mi[2][1];
  res[2][2] = m[2][0] * mi[0][2] + m[2][1] * mi[1][2] + m[2][2] * mi[2][2];


  assert((fabs(res[0][0] - 1.) < MMG5_EPS) &&
         (fabs(res[1][1] - 1.) < MMG5_EPS) &&
         (fabs(res[2][2] - 1.) < MMG5_EPS) && (fabs(res[0][1]) < MMG5_EPS) &&
         (fabs(res[0][2]) < MMG5_EPS) && (fabs(res[1][2]) < MMG5_EPS) &&
         (fabs(res[1][0]) < MMG5_EPS) && (fabs(res[2][0]) < MMG5_EPS) &&
         (fabs(res[2][1]) < MMG5_EPS) && "Matrix inversion");

#endif

  return 1;
}

/**
 * \param dim matrix size.
 * \param m matrix array.
 * \param dm diagonal values array.
 * \param iv array of inverse coreduction basis.
 *
 * Recompose a symmetric matrix from its diagonalization on a simultaneous
 * reduction basis.
 * \warning Eigenvectors in Mmg are stored as matrix rows (the first dimension
 * of the double array spans the number of eigenvectors, the second dimension
 * spans the number of entries of each eigenvector). So the inverse (left
 * eigenvectors) is also stored with transposed indices.
 */
static inline void MMG5_simredmat(int8_t dim, double *m, double *dm, double *iv)
{
  int8_t i, j, k, ij;

  /* Storage of a matrix as a one-dimensional array: dim*(dim+1)/2 entries for
   * a symmetric matrix. */
  ij = 0;

  /* Loop on matrix rows */
  for (i = 0; i < dim; i++)
  {
    /* Loop on the upper-triangular part of the matrix */
    for (j = i; j < dim; j++)
    {
      /* Initialize matrix entry */
      m[ij] = 0.0;
      /* Compute matrix entry as the recomposition of diagonal values after
       * projection on the coreduction basis, using the inverse of the
       * transformation:
       *
       * M_{ij} = \sum_{k,l} V^{-1}_{ki} Lambda_{kl} V^{-1}_{lj} =
       *        = \sum_{k} lambda_{k} V^{-1}_{ki} V^{-1}_{kj}
       *
       * Since the inverse of the transformation is the inverse of an
       * eigenvectors matrix (which is stored in Mmg by columns, and not by
       * rows), the storage of the inverse matrix is also transposed and the
       * indices have to be exchanged when implementing the above formula. */
      for (k = 0; k < dim; k++)
      {
        m[ij] += dm[k] * iv[i * dim + k] * iv[j * dim + k];
      }
      /* Go to the next entry */
      ++ij;
    }
  }
  assert(ij == (dim + 1) * dim / 2);
}

/**
 * \param mesh pointer to the mesh structure.
 * \param m pointer to a \f$(3x3)\f$ metric.
 * \param n pointer to a \f$(3x3)\f$ metric.
 * \param mr computed \f$(3x3)\f$ metric.
 * \return 0 if fail, 1 otherwise.
 *
 * Compute the intersected (3 x 3) metric from metrics \a m and \a n : take
 * simultaneous reduction, and proceed to truncation in sizes.
 *
 */
int MMG5_intersecmet33(double  hmin,
                       double  hmax,
                       double *m,
                       double *n,
                       double *mr)
{
  double vp[3][3], dm[3], dn[3], d[3], ivp[3][3];
  double isqhmin, isqhmax;
  int8_t i;

  isqhmin = 1.0 / (hmin * hmin);
  isqhmax = 1.0 / (hmax * hmax);

  /* Simultaneous reduction */
  if (!MMG5_simred3d(m, n, dm, dn, vp))
    return 0;

  /* Diagonal values of the intersected metric */
  for (i = 0; i < 3; i++)
  {
    d[i] = MG_MAX(dm[i], dn[i]);
    d[i] = MG_MIN(isqhmin, MG_MAX(d[i], isqhmax));
  }

  /* Intersected metric = tP^-1 diag(d0,d1,d2)P^-1, P = (vp0, vp1,vp2) stored in
   * columns */

  /* Compute the inverse of the eigenvectors matrix */
  if (!MMG5_invmat33(vp, ivp))
    return 0;

  /* Recompose matrix */
  MMG5_simredmat(3, mr, d, (double *)ivp);

  return 1;
}

#undef MG_MIN
#undef MG_MAX
