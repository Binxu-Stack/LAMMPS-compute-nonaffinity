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

/* ======================================================================
 *
Author for compute nonaffinity: Bin Xu (xubinrun@gmail.com)

Obtaining hessian matirx: thank Anthony B. Costa, anthony.costa@numericalsolutions.org
http://bitbucket.org/numericalsolutions/lammps-hessian
========================================================================= */

#include "lmptype.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "universe.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "min.h"
#include "modify.h"
#include "fix.h"
#include "output.h"
#include "thermo.h"
#include "finish.h"
#include "kspace.h"
#include "bond.h"
#include "pair.h"
#include "dihedral.h"
#include "timer.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "irregular.h"
#include "compute.h"
#include "stdio.h"
#include "math.h"
#include "neighbor.h"
#include "compute_nonaffinity.h"
#include "atom.h"
#include "atom_vec.h"
#include "error.h"
#include "update.h"
#include "memory.h"
#include "domain.h"
#include "modify.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "comm.h"
#include "math_const.h"

//#define DEBUG

#define MAXLINE 256
#define PI 3.141592653589793238462643383279
using namespace LAMMPS_NS;

#ifdef MKL
#include "mkl.h"
#define dsyev_ dsyev
#else
extern "C" {
    extern void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,double* w, double* work, int* lwork, int* info );
};
#endif


/* ---------------------------------------------------------------------- */

ComputeNonaffinity::ComputeNonaffinity(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg) {
  // command style:
  // compute 1 all nonaffinity epsilon derivative_file nonaffinity_file
  if (narg != 6)
    error->all(FLERR, "Illegal compute hessian command");

  //eps = atof(arg[3]);
  //if(eps < 1e-3) error->all(FLERR,"eps two small!");
  epsilon = atof(arg[3]);
  //epsilon = 1e-11;
  //printf("Epsilon: %g\n",epsilon);
  iepsilon = 1 / epsilon;
  memory->create(derivative_file, strlen(arg[4])+1, "Compute_nonafinity:derivative_file");
  //derivative_file = new char [strlen(arg[4])+1];
  strcpy(derivative_file, arg[4]);
  memory->create(nonaffinity_file, strlen(arg[5])+1, "Compute_nonafinity:nonaffinity_file");
  //nonaffinity_file = new char [strlen(arg[5])+1];
  strcpy(nonaffinity_file, arg[5]);
  //printf("Nonaffinity file %s\n",nonaffinity_file);
  // setup compute
  int id = modify->find_compute("thermo_pe");
  if (id < 0) error->all(FLERR,"Command derivative cannot find thermo_pe compute");
  pe_compute = modify->compute[id];

  id = modify->find_compute("thermo_press");
  if (id < 0) error->all(FLERR,"Command derivative Cannot find thermo_press compute");
  press_compute = modify->compute[id];

  /* even though this is a massive 2d array, return the a vector instead.
   * we will explicitly manage the addressing in each dimension with a
   * preprocessor index macro. */
  vector_flag = 1;
  extvector = 0;

  /* these values will change if the system size changes. */
  ndof = ndofs = atom->natoms * domain->dimension;
  //size_vector = ndofs * ndofs;
  size_vector = 2;

  mylocalsize = 0;
  myglobalsize = 0;

  fglobal_ref = fglobal_new = fglobal_copy = NULL;
  hessian = NULL;
  me = comm->me;
}

/* ---------------------------------------------------------------------- */

ComputeNonaffinity::~ComputeNonaffinity() {
  free(fglobal_ref);
  free(fglobal_new);
  free(fglobal_copy);
  //memory->destroy(hessian);
  free(hessian);
  memory->destroy(x0);
  memory->destroy(derivative_file);
  memory->destroy(nonaffinity_file);
  memory->destroy(eigen);
  memory->destroy(xy_nonaffinity);
  memory->destroy(pxx_x);
  memory->destroy(pyy_x);
  memory->destroy(pzz_x);
  memory->destroy(pxy_x);
  memory->destroy(pxz_x);
  memory->destroy(pyz_x);
}

/* ---------------------------------------------------------------------- */
void ComputeNonaffinity::allocate(){
  memory->create(x0,3*atom->nlocal,"nonaffinity:x0");
  memory->create(xy_nonaffinity,atom->natoms,"nonaffinity:xy_nonaffinity");
  memory->create(pxx_x,ndof,"nonaffinity:pxx_x");
  memory->create(pyy_x,ndof,"nonaffinity:pyy_x");
  memory->create(pzz_x,ndof,"nonaffinity:pzz_x");
  memory->create(pxy_x,ndof,"nonaffinity:pxy_x");
  memory->create(pxz_x,ndof,"nonaffinity:pxz_x");
  memory->create(pyz_x,ndof,"nonaffinity:pyz_x");
}

void ComputeNonaffinity::compute_vector(void) {
  allocate();
  invoked_vector = update->ntimestep;

  /* tags must be defined and consecutive. */
  if (atom->tag_enable == 0)
    error->all(FLERR,
               "Cannot use Hessian compute unless atoms have IDs");
  if (atom->tag_consecutive() == 0)
    error->all(FLERR,
               "Atom IDs must be consecutive for Hessian compute");

  /* get pointers to all the original data. */
  double **x = atom->x;
  double **f = atom->f;

  /* the global force and hessian arrays must be explicitly the correct size. */
  int needglobalsize = atom->natoms;
  int ndofs = atom->natoms * domain->dimension;
#ifdef DEBUG
    printf("Number of atoms: %i\n", atom->natoms);
    printf("ndofs: %i\n", ndofs);
#endif
  //bigint nhessianelements = ndofs * ndofs;
  long long nhessianelements = (long long)ndofs * (long long)ndofs;
  //printf("nhessianelements: %lld, %lld\n", nhessianelements, ndofs*ndofs);
  if (needglobalsize != myglobalsize) {
    free (fglobal_ref); 
    free (fglobal_new); 
    free (fglobal_copy);
    free (hessian);

    fglobal_ref = (double *) malloc (ndofs * sizeof (double));   
    fglobal_new = (double *) malloc (ndofs * sizeof (double));   
    fglobal_copy = (double *) malloc (ndofs * sizeof (double));   
    //memory->create(hessian, nhessianelements, "Compute Hessian: hessian");
    hessian = (double *) malloc (nhessianelements * sizeof (double));
    if (hessian == NULL) {
      error->all(FLERR, "hessian matrix allocation failed.");
    }
#ifdef DEBUG
    printf("Number of elemnets in hessian: %i\n", nhessianelements);
#endif

    /* always be sure to set the output vector since the address keeps changing. */
    //vector = hessian;
    double dummy[2] = {0., 0.};
    vector = dummy;

    myglobalsize = needglobalsize;
  }

  /* a lot of the hessian will be zero, so start there. */
  memset (hessian, 0, nhessianelements * sizeof(double));
  //for (int i = 0; i < nhessianelements; ++i){
  //  hessian[i] = 0.0;
  //}

  /* set up a map if none exists so we can incrementally loop through all dofs
   * regardless of the location of the atom data. */
  int mapflag = 0;
  if (atom->map_style == 0) {
    mapflag = 1;
    atom->map_init();
    atom->map_set();
  }

  /* no energy or virial updates. */
  int eflag = 0;
  int vflag = 0;

  /* allow pair and kspace compute to be turned off via modify flags. */
  if (force->pair && force->pair->compute_flag)
    pair_compute_flag = 1;
  else
    pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag)
    kspace_compute_flag = 1;
  else
    kspace_compute_flag = 0;

  /* do a standard force call to get the reference forces. */
  comm->forward_comm();
  force_clear();
  if (modify->n_pre_force) modify->pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag, vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (kspace_compute_flag) force->kspace->compute(eflag, vflag);
  if (force->newton) comm->reverse_comm();
  if (modify->n_post_force) modify->post_force(vflag);

  /* construct fglobal_ref by explicit scatter and reduce to preserve atom-id
   * ordering. */
  int m, reduce_m;
  memset (&fglobal_copy[0], 0, myglobalsize * domain->dimension * sizeof (double));
  for (int i = 1; i <= atom->natoms; i++) {
    m = atom->map(i);
    if (atom->mask[m]) {
      reduce_m = atom->tag[m] - 1;
      for (int j = 0; j < domain->dimension; j++)
        fglobal_copy[idx2_c(reduce_m, j, atom->natoms)] = f[m][j];
    }
  }
  MPI_Allreduce (fglobal_copy, fglobal_ref, ndofs, MPI_DOUBLE, MPI_SUM, world);

  /* do numerical hessian compute by forward differences. */
  int n, reduce_n, index_a, index_b, global_atom_a, global_atom_b;
  double mass, difference, mass_weight, xstore;
  for (int i = 1; i <= atom->natoms; i++) {

    m = atom->map(i);
    if (atom->mask[m]) {
      /* global ids in lammps are handled by 1-based indexing, while everything
       * local is 0-based. */
      global_atom_a = atom->tag[m] - 1;
      MPI_Bcast(&global_atom_a, 1, MPI_INT, comm->me, world);

      if (atom->rmass) {
        mass = atom->rmass[m];
        MPI_Bcast(&mass, 1, MPI_DOUBLE, comm->me, world);
      } else {
        mass = atom->mass[atom->type[m]];
        MPI_Bcast(&mass, 1, MPI_DOUBLE, comm->me, world);
      }
    }

    for (int j = 0; j < domain->dimension; j++) {
      /* increment the dof by epsilon on the right task. */
      if (atom->mask[m]) {
        xstore = x[m][j];
        x[m][j] += epsilon;
      }

      /* standard force call. */
      comm->forward_comm();
      force_clear();
      if (modify->n_pre_force) modify->pre_force(vflag);

      if (pair_compute_flag) force->pair->compute(eflag, vflag);

      if (atom->molecular) {
        if (force->bond) force->bond->compute(eflag, vflag);
        if (force->angle) force->angle->compute(eflag, vflag);
        if (force->dihedral) force->dihedral->compute(eflag, vflag);
        if (force->improper) force->improper->compute(eflag, vflag);
      }

      if (kspace_compute_flag) force->kspace->compute(eflag, vflag);

      /* put the original position back. */
      if (atom->mask[m]) x[m][j] = xstore;

      if (force->newton) comm->reverse_comm();
      if (modify->n_post_force) modify->post_force(vflag);

      /* construct fglobal_new by explicit scatter and reduce to preserve
       * atom-id ordering. */
      memset (&fglobal_copy[0], 0, myglobalsize * domain->dimension * sizeof (double));
      for (int k = 1; k <= atom->natoms; k++) {
        n = atom->map(k);
        if (atom->mask[n]) {
          reduce_n = atom->tag[n] - 1;
          for (int l = 0; l < domain->dimension; l++)
            fglobal_copy[idx2_c(reduce_n, l, atom->natoms)] = f[n][l];
        }
      }
      MPI_Allreduce (fglobal_copy, fglobal_new, ndofs, MPI_DOUBLE, MPI_SUM, world);

      /* compute the difference (not using symmetry so we can do an in-place
       * reduciton). */
      index_a = j + domain->dimension * global_atom_a;
      for (int k = 1; k <= atom->natoms; k++) {
        n = atom->map(k);
        if (atom->mask[n]) {
          global_atom_b = atom->tag[n] - 1;

          /* don't need to broadcast the second mass because it will only be used
           * on this rank. */
          if (atom->rmass)
            mass_weight = 1 / sqrt(mass * atom->rmass[n]);
          else
            mass_weight = 1 / sqrt(mass * atom->mass[atom->type[n]]);

          /* once again, global arrays use 1-based indexing, so have to rebase
           * them to 0. */
          for (int l = 0; l < domain->dimension; l++) {
            index_b = l + domain->dimension * global_atom_b;
            difference =
                fglobal_ref[idx2_c(global_atom_b, l, atom->natoms)] - \
                fglobal_new[idx2_c(global_atom_b, l, atom->natoms)];

            hessian[idx2_c((long long)index_a, (long long)index_b, (long long)ndofs)] =
                difference * iepsilon * mass_weight;
          }
        }
      }
    }
  }

  /* only reduce the hessian to the root task. */
  MPI_Reduce(MPI_IN_PLACE, hessian, nhessianelements, MPI_DOUBLE, MPI_SUM, 0, world);

  /* do a standard force call to get the original forces back. */
  comm->forward_comm();
  force_clear();
  if (modify->n_pre_force) modify->pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag, vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (kspace_compute_flag) force->kspace->compute(eflag, vflag);
  if (force->newton) comm->reverse_comm();
  if (modify->n_post_force) modify->post_force(vflag);
  solve_eigen();
  solve_derivative();
  solve_nonaffinity();
  /* destroy the atom map. */
  if (mapflag) {
    atom->map_delete();
    atom->map_style = 0;
  }

}

void ComputeNonaffinity::force_clear() {
  size_t nbytes;
  int nlocal = atom->nlocal;

  nbytes = sizeof(double) * nlocal;
  if (force->newton) nbytes += sizeof(double) * atom->nghost;

  if (nbytes) memset (&atom->f[0][0], 0, 3 * nbytes);
}

void ComputeNonaffinity::solve_nonaffinity(){
  double volume;
  if (domain->dimension == 2){
    volume = domain->xprd * domain->yprd;
  }else{
    volume = domain->xprd * domain->yprd * domain->zprd;
  }
  char  str[MAXLINE];
  FILE *out_nonaffinity;
  if (me == 0){
    out_nonaffinity = fopen(nonaffinity_file,"w");
    if (out_nonaffinity == NULL){
      sprintf(str, "Cannot open nonaffinity file: %s for writing", nonaffinity_file);
      error->all(FLERR, str);
    }
  }
  /* Solve the orientation of each mode */
  // NOTE: only two dimension is considered currently
  // calculate the maximum nonaffine shear modulus and softest orientation for each mode
  double * max_nonaffine_modulus_modes;
  //double * max_nonaffine_disp_modes;
  double * theta_softest;
  memory->create(max_nonaffine_modulus_modes, ndof, "nonaffine_modulus_modes");
  memory->create(theta_softest, ndof, "theta_softest");
  for (int i = 0; i < ndof; ++i){
    double tau_max_square, tmp_softest, tau_max,tan_2theta_max;
    tau_max_square = (pxx_x[i]-pyy_x[i])*(pxx_x[i]-pyy_x[i])*0.25+pxy_x[i]*pxy_x[i];
    max_nonaffine_modulus_modes[i] = volume / eigen[i] * tau_max_square;
    //max_nonaffine_disp_modes[i] = volume / eigen[i] * sqrt(tau_max_square);
    tan_2theta_max = -(pxx_x[i]-pyy_x[i])*0.5/pxy_x[i];
    tmp_softest = atan(tan_2theta_max)*0.5;
    // confine orientation in region [0,pi]
    if(tmp_softest < 0.0) tmp_softest += PI*0.5;
    tau_max = (pxx_x[i]-pyy_x[i])*0.5*sin(2.*tmp_softest)-pxy_x[i]*cos(2.*tmp_softest);
    // Choose the one softest diection in the two soft direction
    if (f_xxx[i]*tau_max>0.){
      theta_softest[i] = tmp_softest;
    }else{
      theta_softest[i] = tmp_softest + PI*0.5;
    }
  }

  // find our the mode that dominate the nonaffine shear modulus of each atom
  int * max_index; // dominate mode index
  double * max_value; // max_nonaffine_modulus_modes of dominate mode
  int natoms = atom->natoms;
  memory->create(max_index, natoms, "max_index");
  memory->create(max_value, natoms, "max_value");
  for (int i = 0; i < natoms; ++i) {
    max_value[i] = 0.0;
  }
  double tmp;
  double tmp2;
  //for (int i = domain->dimension; i < ndof; ++i) {
  for (int i = 0; i < ndof; ++i) {
    for (int j = 0; j < natoms; ++j) {
      // only for 2D
      // maximum nonaffine modulus contribution of one mode
      tmp = hessian[i*ndof+j*2]*hessian[i*ndof+j*2]*max_nonaffine_modulus_modes[i] + hessian[i*ndof+j*2+1]*hessian[i*ndof+j*2+1]*max_nonaffine_modulus_modes[i];
      if (tmp > max_value[j]){
	max_value[j] = tmp;
	max_index[j] = i;
      }
    }
  }

  // Print out softest shear orientation of each atom
  FILE *out_orientation;
  out_orientation = fopen("Orientation.dat","w");
  fprintf(out_orientation,"# theta (rad) index_of_mode\n");
  for (int i = 0; i < natoms; ++ i) {
    double theta,second_theta,first,second,ratio;
    theta = theta_softest[max_index[i]];
    fprintf(out_orientation,"%16g %i\n",theta, max_index[i]);
  }
  fclose(out_orientation);


  // print header of nonaffinity file
  if (me == 0){
    fprintf(out_nonaffinity, "# atom_id xy_nonaffinity\n");
  }
  for (int i=0; i < atom->natoms; ++i){
    xy_nonaffinity[i] = 0.0;
  }
  for (int i=0; i < ndof; ++i){
    for (int j = 0; j < ndof; ++j){
      int index = static_cast<int>(j/domain->dimension);
      //xy_nonaffinity[index] += hessian[i*ndof+j]*hessian[i*ndof+j]*-pxy_x[i]*pxy_x[i]/eigen[i];
      // tricks here for configuration close to saddle
      if (fabs(eigen[i]) > 1e-11)
      	xy_nonaffinity[index] += hessian[i*ndof+j]*hessian[i*ndof+j]*-pxy_x[i]*pxy_x[i]/fabs(eigen[i]);
    }
  }
  for (int i=0; i < atom->natoms; ++i){
    fprintf(out_nonaffinity, "%i %10g\n",i+1, xy_nonaffinity[i]*volume);
  }
  fclose(out_nonaffinity);

}
void ComputeNonaffinity::solve_derivative(){
  int nlocal = atom->nlocal;
  char str[MAXLINE];
  double *press_0 = new double[6];
  double *e = new double[5];
  double **press; 
  double press_x[6] = {0.};
  memory->create(press, 5, 6, "NONaffine_press");
  memory->create(f_xxx,ndof, "Compute_nonaffinity:f_xxx");
  double * xvec = atom->x[0];
  e[0] = energy_press(0);
  double *tmp = press_compute->vector;
  for (int i = 0; i < 6; ++i){
    press_0[i] = tmp[i];
    //printf("%i %f\n", i,press_0[i]);
  }
  for (int i = 0; i < nlocal*3; ++i) {
    x0[i] = xvec[i];
  }
  FILE *out_deri;
  if (me == 0){
    out_deri = fopen(derivative_file,"w");
    if (out_deri == NULL){
      sprintf(str, "Cannot open derivative file: %s for writing", derivative_file);
      error->all(FLERR, str);
    }
  }
  // print header of derivative file
  if (me == 0){
    fprintf(out_deri, "# index eigen f_xx f_xxx pxx_x pyy_x pzz_x pxy_x pxz_x pyz_x\n");
  }
  int ilocal;
  tagint tag;
  double eps = 2.5e-6*atom->natoms; // Good in my case
  //double eps = 1.0e-2;
  double ieps = 1.0/eps;
  double diff[5] = {0., -2.*eps, -1.*eps, 1.*eps, 2.*eps};
  double f_xx;
  for (int i = 0; i < ndof; ++i){
    for (int index = 1; index < 5; ++index){
      for (int j = 0; j < ndof; ++j){
        // get the id
        tag = static_cast<tagint>(j/domain->dimension+1);
        // get the index in atom->x
        ilocal = atom->map(tag);
        if (ilocal >=0 && ilocal < nlocal){
  	  int n = ilocal*3;
  	  int loc = n + j%domain->dimension;
  	  xvec[loc] = x0[loc] + hessian[i*ndof+j]*diff[index];
        }
      }
      e[index] = energy_press(0);
      double * tmp = press_compute->vector;
      for (int ii = 0; ii < 6; ++ii){
	press[index][ii] = tmp[ii];
      }
    }
    // solve third order derivative
#define i_m2 1
#define i_m1 2
#define i_0  0
#define i_1  3
#define i_2  4
    f_xxx[i] = (-1.*e[i_m2]+2.*e[i_m1]+0.*e[i_0]-2.*e[i_1]+e[i_2])/(2.*eps*eps*eps);
    f_xx = (-1.*e[i_m2]+16.*e[i_m1]-30.*e[i_0]+16.*e[i_1]-1.*e[i_2])/(12.*eps*eps);
    // solve stress gradient
    //  http://web.media.mit.edu/~crtaylor/calculator.html
    //  f_x = (-11*f[i+0]+18*f[i+1]-9*f[i+2]+2*f[i+3])/(6*1.0*h**1)
    for (int ii = 0; ii < 6; ++ii) {
      //press_x[i] = (-11.*press[0][i]+18.*press[1][i]-9.*press[2][i]+2.*press[3][i])/(6.*eps);
      // f_x = (1*f[i-2]-8*f[i-1]+0*f[i+0]+8*f[i+1]-1*f[i+2])/(12*1.0*h**1)
      press_x[ii] = (press[i_m2][ii]-8.*press[i_m1][ii]+8.*press[i_1][ii]-1.*press[i_2][ii])/(12.*eps);
    }
    pxx_x[i] = press_x[0];
    pyy_x[i] = press_x[1];
    pzz_x[i] = press_x[2];
    pxy_x[i] = press_x[3];
    pxz_x[i] = press_x[4];
    pyz_x[i] = press_x[5];
    fprintf(out_deri,"%i %10g %10g %10g %10g %10g %10g %10g %10g %10g\n", 
  	i, eigen[i], f_xx, f_xxx[i], pxx_x[i], pyy_x[i], pzz_x[i], pxy_x[i], pxz_x[i], pyz_x[i]);
  }
  delete []press_0;
  delete []e;
  memory->destroy(press);
  // restore to the initial condition
  for(int ii = 0; ii < nlocal*3; ++ii){
    xvec[ii] = x0[ii];
  }
  fclose(out_deri);
}


void ComputeNonaffinity::solve_eigen(){
  int ndof = atom->natoms * domain->dimension;
  int n = ndof;
  int lda = n, info, lwork;
  double wkopt;
  //double *w = new double[n];
  //#eigen = new double[n];
  memory->create(eigen,n,"Compute_nonaffinity:eigen");
  double *work, *z;
  lwork = -1;
  if (screen) fprintf(screen, "Now try to solve eigens...\n");
  /* Query and allocate the optimal workspace */
  dsyev_("Vectors", "Upper", &n, hessian, &lda, eigen, &wkopt, &lwork, &info);
  lwork = int(wkopt);
  //work = (double*)malloc( lwork*sizeof(double) );
  memory->create(work,lwork, "Compute_nonaffinity:work");
  /* Solve eigenproblem */
  dsyev_( "Vectors", "Upper", &n, hessian, &lda, eigen, work, &lwork, &info );
  /* Check for convergence */
  if (info > 0){
    error->all(FLERR, "The algorithm failed to compute eigenvalues." );
  }
  //FILE *fp;
  //const int max_vector_num = ndof; 
  //int max_num = MIN(max_vector_num, ndof);
  //fp = fopen("eigen.dat","w");
  //for (int i = 0; i < max_num; ++i) {
  //  fprintf(fp, "%f\n",w[i]);
  //}
  //fclose(fp);
  //fp = fopen("eigen_vector.dat","w");
  //fprintf(fp, "%i\n", max_num);
  //for (int i = 0; i < max_num; ++i) {
  //  for (int j = 0; j < ndof; ++j) fprintf(fp, "%g\n",hessian[i*ndof+j]);
  //}
  //fclose(fp);
  memory->destroy(work);

}
double ComputeNonaffinity::energy_press(int force_reneighbor){
  int eflag = 1;
  int vflag = 1;

  if (force_reneighbor) {
    double **x = atom->x;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; ++i) domain->remap(x[i],image[i]);
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->reset_box();
    Irregular *irregular = new Irregular(lmp);
    irregular->migrate_atoms(1);
    delete irregular;
    if (domain->triclinic) domain->lamda2x(atom->nlocal);
  }
  int nflag = neighbor->decide();
  if (nflag == 0){
    timer->stamp();
    comm->forward_comm();
    timer->stamp(Timer::COMM);
  }else {
    domain->pbc();
    if (domain->box_change) {
      domain->reset_box();
      comm->setup();
      if (neighbor->style) neighbor->setup_bins();
    }
    timer->stamp();
    comm->exchange();
    if (atom->sortfreq > 0 &&
        update->ntimestep >= atom->nextsort) atom->sort();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    timer->stamp(Timer::COMM);
    neighbor->build(1);
  }

  if (force->pair && force->pair->compute_flag)
    pair_compute_flag = 1;
  else
    pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag)
    kspace_compute_flag = 1;
  else
    kspace_compute_flag = 0;

  /* do a standard force call to get the reference forces. */
  comm->forward_comm();
  force_clear();
  if (modify->n_pre_force) modify->pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag, vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (kspace_compute_flag) force->kspace->compute(eflag, vflag);
  if (force->newton) comm->reverse_comm();
  if (modify->n_post_force) modify->post_force(vflag);
  // compute potential energy of system
  // normalize if thermo PE does

  double energy = pe_compute->compute_scalar();
  press_compute->compute_vector();

  // if reneighbored, atoms migrated
  // if resetflag = 1, update x0 of atoms crossing PBC
  // reset vectors used by lo-level minimizer


  return energy;

}


