/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(nonaffinity, ComputeNonaffinity)

#else

#ifndef LMP_COMPUTE_NONAFFINITY_H
#define LMP_COMPUTE_NONAFFINITY_H

#include "compute.h"

/* zero-based row- and column-major indexing macros for the hessian. */
#define idx2_r(i, j, ldj) ((i * ldj) + j)
#define idx2_c(i, j, ldi) ((j * ldi) + i)

namespace LAMMPS_NS {

  class ComputeNonaffinity : public Compute {
  public:
    ComputeNonaffinity(class LAMMPS *, int, char **);
    ~ComputeNonaffinity();
    void init() {}
    void compute_vector();
  
  protected:
    int mylocalsize;
    int myglobalsize;
  
    double *fglobal_ref, *fglobal_new, *fglobal_copy;
    double *hessian;
  
    double epsilon, iepsilon;
    //double eps;
  
    int pair_compute_flag;
    int kspace_compute_flag;
  
    void force_clear();
    void solve_eigen();
    void solve_derivative();
    void solve_nonaffinity();

  private:
    int me;
    int ndof;
    int ndofs;
    char* derivative_file;
    char* nonaffinity_file;
    double * eigen;
    double * f_xxx;
    double * xy_nonaffinity;
    double * pxx_x;
    double * pyy_x;
    double * pzz_x;
    double * pxy_x;
    double * pxz_x;
    double * pyz_x;
    double * press_x;
    double *x0;
    double ** press;
    void allocate();
    void solve_stress_gradient();
    double energy_press(int);
    class Compute *pe_compute;
    class Compute *press_compute;
  };

}

#endif
#endif
