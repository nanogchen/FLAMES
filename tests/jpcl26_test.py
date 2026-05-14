# 
# Copyright (C) Guang Chen et al.
# 
# This file is part of FLAMES program
#
# FLAMES is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FLAMES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#

import sys
import numpy as np
import MDAnalysis as mda

sys.path.insert(0, "../")
from flames.q_gen import get_rho_q_noFF, get_q_points_all_quads, get_binning_averages

def get_static_sf_mol(q_points, system, traj, NatomPerMol):

	"""
	get static structure factor S(q_vec) at given time using mol COM
	"""
	n_qpoints = len(q_points)
	ssf = np.zeros((n_qpoints, len(traj)))
	Nmol = int(system.atoms.n_atoms/NatomPerMol) 

	ifr=0    
	for _ in traj:

		coords = system.positions
		coords_mol = np.reshape(coords, (Nmol, NatomPerMol, 3))
		coords_COM = np.mean(coords_mol, axis=1)
    
		rho_q = get_rho_q_noFF(coords_COM, q_points)
		sq_t = np.real(rho_q*rho_q.conjugate()) 

		ssf[:,ifr] = sq_t
		ifr+=1

	return ssf/Nmol

# -----------------------------------------------------------------------------

if __name__ == '__main__':

	u = mda.Universe('md_0_50.gro',"md_50_100_BCwhole.xtc") # use own trajectory with pbc whole
	system = u.select_atoms("all")
	bx,by,bz=u.dimensions[:3]
	
	# undecane
	q_end = 1.0
	max_points = 3000
	num_q_bins = 50
	Fr_start = 0
	Fr_stop = 2500
	Fr_step = 5
	NatomPerMol = 11

	q_points = get_q_points_all_quads(np.array([bx, by, bz]), q_end, max_points=max_points)
	ssf = get_static_sf_mol(q_points, system, u.trajectory[Fr_start:Fr_stop+1:Fr_step], NatomPerMol)
	qr, ssf_qr = get_binning_averages(num_q_bins, q_end, ssf, q_points)

