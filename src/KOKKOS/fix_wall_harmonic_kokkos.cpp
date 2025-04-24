// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Mitch Murphy (alphataubio@gmail.com)
------------------------------------------------------------------------- */

#include "fix_wall_harmonic_kokkos.h"

#include "atom_masks.h"
#include "atom_kokkos.h"
#include "error.h"
#include "input.h"
#include "memory_kokkos.h"

#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
FixWallHarmonicKokkos<DeviceType>::FixWallLHarmonicKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixWallHarmonic(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | V_MASK | MASK_MASK;
  datamask_modify = F_MASK;

  memoryKK->create_kokkos(k_cutoff,6,"wall_harmonic:cutoff");
  memoryKK->create_kokkos(k_epsilon,6,"wall_harmonic:epsilon");

  d_cutoff = k_cutoff.template view<DeviceType>();
  d_epsilon = k_epsilon.template view<DeviceType>();
}

template<class DeviceType>
FixWallHarmonicKokkos<DeviceType>::~FixWallHarmonicKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_cutoff);
  memoryKK->destroy_kokkos(k_epsilon);
  memoryKK->destroy_kokkos(k_vatom,vatom);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixWallHarmonicKokkos<DeviceType>::precompute(int m)
{
  FixWallHarmonic::precompute(m);

  for( int i=0 ; i<6 ; i++ ) {
    k_cutoff.h_view(i) = cutoff[i];
    k_epsilon.h_view(i) = epsilon[i];
  }

  k_cutoff.template modify<LMPHostType>();
  k_epsilon.template modify<LMPHostType>();

  k_cutoff.template sync<DeviceType>();
  k_epsilon.template sync<DeviceType>();

}


/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixWallHarmonicKokkos<DeviceType>::post_force(int vflag)
{

  // reallocate per-atom arrays if necessary

  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"wall_harmonic:vatom");
    d_vatom = k_vatom.template view<DeviceType>();
  }

  FixWallHarmonic::post_force(vflag);

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }
}

/* ----------------------------------------------------------------------
   interaction of all particles in group with a wall
   m = index of wall coeffs
   which = xlo,xhi,ylo,yhi,zlo,zhi
   error if any particle is on or behind wall
------------------------------------------------------------------------- */

template <class DeviceType>
void FixWallHarmonicKokkos<DeviceType>::wall_particle(int m_in, int which, double coord_in)
{
  m = m_in;
  coord = coord_in;

  atomKK->sync(execution_space,datamask_read);
  d_x = atomKK->k_x.template view<DeviceType>();
  d_f = atomKK->k_f.template view<DeviceType>();
  d_mask = atomKK->k_mask.template view<DeviceType>();
  int nlocal = atomKK->nlocal;

  dim = which / 2;
  side = which % 2;
  if (side == 0) side = -1;

  double result[13] = {0.0};

  copymode = 1;
  Kokkos::parallel_reduce(nlocal,*this,result);
  copymode = 0;

  ewall[0] += result[0];
  ewall[m+1] += result[m+1];
  atomKK->modified(execution_space,datamask_modify);

  if (vflag_global) {
    virial[0] += result[7];
    virial[1] += result[8];
    virial[2] += result[9];
    virial[3] += result[10];
    virial[4] += result[11];
    virial[5] += result[12];
  }
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixWallHarmonicKokkos<DeviceType>::operator()(const int &i, value_type result) const {
  if (d_mask(i) & groupbit) {
    double delta;
    if (side < 0) delta = d_x(i,dim) - coord;
    else delta = coord - d_x(i,dim);
    if (delta >= d_cutoff(m)) return;
    if (delta <= 0.0)
      Kokkos::abort("Particle on or inside fix wall surface");

    double dr = cutoff[m] - delta;
    double fwall = side * 2.0 * epsilon[m] * dr;
    d_f(i,dim) -= fwall;
    result[0] +=  epsilon[m] * dr * dr;
    result[m+1] += fwall;

    if (evflag) {
      double vn;
      if (side < 0)
        vn = -fwall * delta;
      else
        vn = fwall * delta;

      v_tally(result, dim, i, vn);
    }
  }
}

/* ----------------------------------------------------------------------
   tally virial component into global and per-atom accumulators
   n = index of virial component (0-5)
   i = local index of atom
   vn = nth component of virial for the interaction
   increment nth component of global virial by vn
   increment nth component of per-atom virial by vn
   this method can be used when fix computes forces in post_force()
   and the force depends on a distance to some external object
     e.g. fix wall/lj93: compute virial only on owned atoms
------------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixWallHarmonicKokkos<DeviceType>::v_tally(value_type result, int n, int i, double vn) const
{
  if (vflag_global)
    result[n+7] += vn;

  if (vflag_atom)
    Kokkos::atomic_add(&(d_vatom(i,n)),vn);
}

namespace LAMMPS_NS {
template class FixWallHarmonicKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixWallHarmonicKokkos<LMPHostType>;
#endif
}
