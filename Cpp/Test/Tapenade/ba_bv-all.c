/*        Generated by TAPENADE     (INRIA, Tropics team)
Tapenade 3.10 (r5498) - 20 Jan 2015 09:48
*/
#include "ba_bv.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
/*  Hint: NB_DIRS_REPROJ_BV should be the maximum number of differentiation directions
*/

/*
Differentiation of rodrigues_rotate_point in reverse (adjoint) mode:
gradient     of useful results: *rot *rotatedPt
with respect to varying inputs: *rot *pt
Plus diff mem management of: rot:in rotatedPt:in pt:in
*/
void rodrigues_rotate_point_bv(double *rot, double(*rotb)[NB_DIRS_REPROJ_BV], double
  *pt, double(*ptb)[NB_DIRS_REPROJ_BV], double *rotatedPt, double(*rotatedPtb)
  [NB_DIRS_REPROJ_BV], int nbdirs) {
  int i;
  double theta, costheta, sintheta, theta_inverse, w[3], w_cross_pt[3], tmp;
  double thetab[NB_DIRS_REPROJ_BV], costhetab[NB_DIRS_REPROJ_BV], sinthetab[NB_DIRS_REPROJ_BV],
    theta_inverseb[NB_DIRS_REPROJ_BV], wb[3][NB_DIRS_REPROJ_BV], w_cross_ptb[3][NB_DIRS_REPROJ_BV],
    tmpb[NB_DIRS_REPROJ_BV];
  int nd;
  double tempb[NB_DIRS_REPROJ_BV];
  int ii1;
  // norm of rot
  theta = 0.;
  for (i = 0; i < 3; ++i)
    theta = theta + rot[i] * rot[i];
  pushreal8(theta);
  theta = sqrt(theta);
  costheta = cos(theta);
  sintheta = sin(theta);
  theta_inverse = 1.0 / theta;
  w[0] = rot[0] * theta_inverse;
  w[1] = rot[1] * theta_inverse;
  w[2] = rot[2] * theta_inverse;
  w_cross_pt[0] = w[1] * pt[2] - w[2] * pt[1];
  w_cross_pt[1] = w[2] * pt[0] - w[0] * pt[2];
  w_cross_pt[2] = w[0] * pt[1] - w[1] * pt[0];
  tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2])*(1. - costheta);
  for (nd = 0; nd < nbdirs; ++nd) {
    for (ii1 = 0; ii1 < 3; ++ii1)
      w_cross_ptb[ii1][nd] = 0.0;
    for (ii1 = 0; ii1 < 3; ++ii1)
      wb[ii1][nd] = 0.0;
    ptb[2][nd] = ptb[2][nd] + costheta*rotatedPtb[2][nd];
    costhetab[nd] = pt[2] * rotatedPtb[2][nd];
    w_cross_ptb[2][nd] = w_cross_ptb[2][nd] + sintheta*rotatedPtb[2][nd];
    sinthetab[nd] = w_cross_pt[2] * rotatedPtb[2][nd];
    wb[2][nd] = wb[2][nd] + tmp*rotatedPtb[2][nd];
    tmpb[nd] = w[2] * rotatedPtb[2][nd];
    rotatedPtb[2][nd] = 0.0;
    ptb[1][nd] = ptb[1][nd] + costheta*rotatedPtb[1][nd];
    costhetab[nd] = costhetab[nd] + pt[1] * rotatedPtb[1][nd];
    w_cross_ptb[1][nd] = w_cross_ptb[1][nd] + sintheta*rotatedPtb[1][nd];
    sinthetab[nd] = sinthetab[nd] + w_cross_pt[1] * rotatedPtb[1][nd];
    wb[1][nd] = wb[1][nd] + tmp*rotatedPtb[1][nd];
    tmpb[nd] = tmpb[nd] + w[1] * rotatedPtb[1][nd];
    rotatedPtb[1][nd] = 0.0;
    ptb[0][nd] = ptb[0][nd] + costheta*rotatedPtb[0][nd];
    costhetab[nd] = costhetab[nd] + pt[0] * rotatedPtb[0][nd];
    w_cross_ptb[0][nd] = w_cross_ptb[0][nd] + sintheta*rotatedPtb[0][nd];
    sinthetab[nd] = sinthetab[nd] + w_cross_pt[0] * rotatedPtb[0][nd];
    wb[0][nd] = wb[0][nd] + tmp*rotatedPtb[0][nd];
    tmpb[nd] = tmpb[nd] + w[0] * rotatedPtb[0][nd];
    tempb[nd] = (1. - costheta)*tmpb[nd];
    wb[0][nd] = wb[0][nd] + pt[0] * tempb[nd];
    ptb[0][nd] = ptb[0][nd] + w[0] * tempb[nd];
    wb[1][nd] = wb[1][nd] + pt[1] * tempb[nd];
    ptb[1][nd] = ptb[1][nd] + w[1] * tempb[nd];
    wb[2][nd] = wb[2][nd] + pt[2] * tempb[nd];
    ptb[2][nd] = ptb[2][nd] + w[2] * tempb[nd];
    costhetab[nd] = costhetab[nd] - (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2])*
      tmpb[nd];
    wb[0][nd] = wb[0][nd] + pt[1] * w_cross_ptb[2][nd];
    ptb[1][nd] = ptb[1][nd] + w[0] * w_cross_ptb[2][nd];
    wb[1][nd] = wb[1][nd] - pt[0] * w_cross_ptb[2][nd];
    ptb[0][nd] = ptb[0][nd] - w[1] * w_cross_ptb[2][nd];
    w_cross_ptb[2][nd] = 0.0;
    wb[2][nd] = wb[2][nd] + pt[0] * w_cross_ptb[1][nd];
    ptb[0][nd] = ptb[0][nd] + w[2] * w_cross_ptb[1][nd];
    wb[0][nd] = wb[0][nd] - pt[2] * w_cross_ptb[1][nd];
    ptb[2][nd] = ptb[2][nd] - w[0] * w_cross_ptb[1][nd];
    w_cross_ptb[1][nd] = 0.0;
    wb[1][nd] = wb[1][nd] + pt[2] * w_cross_ptb[0][nd];
    ptb[2][nd] = ptb[2][nd] + w[1] * w_cross_ptb[0][nd];
    wb[2][nd] = wb[2][nd] - pt[1] * w_cross_ptb[0][nd];
    ptb[1][nd] = ptb[1][nd] - w[2] * w_cross_ptb[0][nd];
    rotb[2][nd] = rotb[2][nd] + theta_inverse*wb[2][nd];
    theta_inverseb[nd] = rot[2] * wb[2][nd];
    wb[2][nd] = 0.0;
    rotb[1][nd] = rotb[1][nd] + theta_inverse*wb[1][nd];
    theta_inverseb[nd] = theta_inverseb[nd] + rot[1] * wb[1][nd];
    wb[1][nd] = 0.0;
    rotb[0][nd] = rotb[0][nd] + theta_inverse*wb[0][nd];
    theta_inverseb[nd] = theta_inverseb[nd] + rot[0] * wb[0][nd];
    thetab[nd] = cos(theta)*sinthetab[nd] - sin(theta)*costhetab[nd] -
      theta_inverseb[nd] / (theta*theta);
  }
  popreal8(&theta);
  for (nd = 0; nd < nbdirs; ++nd)
    if (theta == 0.0)
      thetab[nd] = 0.0;
    else
      thetab[nd] = thetab[nd] / (2.0*sqrt(theta));
  for (i = 2; i > -1; --i)
    for (nd = 0; nd < nbdirs; ++nd)
      rotb[i][nd] = rotb[i][nd] + 2 * rot[i] * thetab[nd];
}
/*  Hint: NB_DIRS_REPROJ_BV should be the maximum number of differentiation directions
*/


/*
Differentiation of radial_distort in reverse (adjoint) mode:
gradient     of useful results: *rad_params *proj
with respect to varying inputs: *rad_params *proj
Plus diff mem management of: rad_params:in proj:in
*/
void radial_distort_bv(double *rad_params, double(*rad_paramsb)[NB_DIRS_REPROJ_BV],
  double *proj, double(*projb)[NB_DIRS_REPROJ_BV], int nbdirs) {
  double rsq, L;
  double rsqb[NB_DIRS_REPROJ_BV], Lb[NB_DIRS_REPROJ_BV];
  int nd;
  rsq = proj[0] * proj[0] + proj[1] * proj[1];
  L = 1 + rad_params[0] * rsq + rad_params[1] * rsq*rsq;
  pushreal8(proj[0]);
  pushreal8(proj[1]);
  proj[0] = proj[0] * L;
  for (nd = 0; nd < nbdirs; ++nd) {
    Lb[nd] = proj[1] * projb[1][nd];
    projb[1][nd] = L*projb[1][nd];
  }
  popreal8(&proj[1]);
  popreal8(&proj[0]);
  for (nd = 0; nd < nbdirs; ++nd) {
    Lb[nd] = Lb[nd] + proj[0] * projb[0][nd];
    projb[0][nd] = L*projb[0][nd];
    rad_paramsb[0][nd] = rad_paramsb[0][nd] + rsq*Lb[nd];
    rsqb[nd] = (rad_params[1] * 2 * rsq + rad_params[0])*Lb[nd];
    rad_paramsb[1][nd] = rad_paramsb[1][nd] + rsq*rsq*Lb[nd];
    projb[0][nd] = projb[0][nd] + 2 * proj[0] * rsqb[nd];
    projb[1][nd] = projb[1][nd] + 2 * proj[1] * rsqb[nd];
  }
}
/*  Hint: NB_DIRS_REPROJ_BV should be the maximum number of differentiation directions
*/


/*
Differentiation of project in reverse (adjoint) mode:
gradient     of useful results: *proj
with respect to varying inputs: *cam *X
Plus diff mem management of: cam:in X:in proj:in-out
*/
void project_bv(double *cam, double(*camb)[NB_DIRS_REPROJ_BV], double *X, double(*Xb
  )[NB_DIRS_REPROJ_BV], double *proj, double(*projb)[NB_DIRS_REPROJ_BV], int nbdirs) {
  int i, k;
  double *C;
  double(*Cb)[NB_DIRS_REPROJ_BV];
  double Xo[3], Xcam[3];
  double Xob[3][NB_DIRS_REPROJ_BV], Xcamb[3][NB_DIRS_REPROJ_BV];
  int nd;
  double tempb0[NB_DIRS_REPROJ_BV];
  double tempb[NB_DIRS_REPROJ_BV];
  int ii1;
  for (i = 0; i < 3; i++)
    for (k = 0; k < nbdirs; k++)
    {
      Xob[i][k] = 0.;
      Xcamb[i][k] = 0.;
    }
  Cb = &camb[3];
  C = &cam[3];
  Xo[0] = X[0] - C[0];
  Xo[1] = X[1] - C[1];
  Xo[2] = X[2] - C[2];
  rodrigues_rotate_point(&cam[0], Xo, Xcam);
  proj[0] = Xcam[0] / Xcam[2];
  proj[1] = Xcam[1] / Xcam[2];
  pushreal8(proj[0]);
  pushreal8(proj[1]);
  radial_distort(&cam[9], proj);
  pushreal8(proj[0]);
  pushreal8(proj[1]);
  proj[0] = proj[0] * cam[6] + cam[7];
  proj[1] = proj[1] * cam[6] + cam[8];
  popreal8(&proj[1]);
  popreal8(&proj[0]);
  for (nd = 0; nd < nbdirs; ++nd) {
    camb[6][nd] = camb[6][nd] + proj[1] * projb[1][nd];
    camb[8][nd] = camb[8][nd] + projb[1][nd];
    projb[1][nd] = cam[6] * projb[1][nd];
  }
  for (nd = 0; nd < nbdirs; ++nd) {
    camb[6][nd] = camb[6][nd] + proj[0] * projb[0][nd];
    camb[7][nd] = camb[7][nd] + projb[0][nd];
    projb[0][nd] = cam[6] * projb[0][nd];
  }
  popreal8(&proj[1]);
  popreal8(&proj[0]);
  radial_distort_bv(&cam[9], &camb[9], proj, projb, nbdirs);
  for (nd = 0; nd < nbdirs; ++nd) {
    for (ii1 = 0; ii1 < 3; ++ii1)
      Xcamb[ii1][nd] = 0.0;
    tempb[nd] = projb[1][nd] / Xcam[2];
    Xcamb[1][nd] = Xcamb[1][nd] + tempb[nd];
    Xcamb[2][nd] = Xcamb[2][nd] - Xcam[1] * tempb[nd] / Xcam[2];
    projb[1][nd] = 0.0;
    tempb0[nd] = projb[0][nd] / Xcam[2];
    Xcamb[0][nd] = Xcamb[0][nd] + tempb0[nd];
    Xcamb[2][nd] = Xcamb[2][nd] - Xcam[0] * tempb0[nd] / Xcam[2];
  }
  rodrigues_rotate_point_bv(&cam[0], &camb[0], Xo, Xob, Xcam, Xcamb, nbdirs)
    ;
  for (nd = 0; nd < nbdirs; ++nd) {
    Xb[2][nd] = Xb[2][nd] + Xob[2][nd];
    Cb[2][nd] = Cb[2][nd] - Xob[2][nd];
    Xob[2][nd] = 0.0;
    Xb[1][nd] = Xb[1][nd] + Xob[1][nd];
    Cb[1][nd] = Cb[1][nd] - Xob[1][nd];
    Xob[1][nd] = 0.0;
    Xb[0][nd] = Xb[0][nd] + Xob[0][nd];
    Cb[0][nd] = Cb[0][nd] - Xob[0][nd];
  }
}
/*  Hint: NB_DIRS_REPROJ_BV should be the maximum number of differentiation directions
*/


/*
Differentiation of computeReprojError in reverse (adjoint) mode:
gradient     of useful results: *err
with respect to varying inputs: *err *w *cam *X
RW status of diff variables: *err:in-out *w:out *cam:out *X:out
Plus diff mem management of: err:in w:in cam:in X:in
*/
void computeReprojError_bv(double *cam, double(*camb)[NB_DIRS_REPROJ_BV], double *X,
  double(*Xb)[NB_DIRS_REPROJ_BV], double *w, double(*wb)[NB_DIRS_REPROJ_BV], double
  feat_x, double feat_y, double *err, double(*errb)[NB_DIRS_REPROJ_BV], int
  nbdirs) {
  double proj[2];
  double projb[2][NB_DIRS_REPROJ_BV];
  int nd;
  int ii1;
  pushreal8(*proj);
  project(cam, X, proj);
  for (nd = 0; nd < nbdirs; ++nd) {
    for (ii1 = 0; ii1 < 2; ++ii1)
      projb[ii1][nd] = 0.0;
    (*wb)[nd] = (proj[1] - feat_y)*errb[1][nd];
    projb[1][nd] = projb[1][nd] + (*w)*errb[1][nd];
    errb[1][nd] = 0.0;
    (*wb)[nd] = (*wb)[nd] + (proj[0] - feat_x)*errb[0][nd];
    projb[0][nd] = projb[0][nd] + (*w)*errb[0][nd];
    errb[0][nd] = 0.0;
  }
  popreal8(proj);
  project_bv(cam, camb, X, Xb, proj, projb, nbdirs);
}

/*
Differentiation of computeFocalPriorError in reverse (adjoint) mode:
gradient     of useful results: *err
with respect to varying inputs: *err *cam1 *cam2 *cam3
RW status of diff variables: *err:in-zero *cam1:out *cam2:out *cam3:out
Plus diff mem management of: err:in cam1:in cam2:in cam3:in
*/
void computeFocalPriorError_b(double *cam1, double *cam1b, double *cam2,
  double *cam2b, double *cam3, double *cam3b, double *err, double *errb)
{
  cam1b[6] = cam1b[6] + *errb;
  cam2b[6] = cam2b[6] - 2 * (*errb);
  cam3b[6] = cam3b[6] + *errb;
  *errb = 0.0;
}

/*
Differentiation of computeZachWeightError in reverse (adjoint) mode:
gradient     of useful results: *err
with respect to varying inputs: *err *w
RW status of diff variables: *err:in-zero *w:out
Plus diff mem management of: err:in w:in
*/
void computeZachWeightError_b(double *w, double *wb, double *err, double *errb
  ) {
  *wb = -(2 * (*w)*(*errb));
  *errb = 0.0;
}