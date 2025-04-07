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

import math
import numpy as np
import MDAnalysis as mda

import numba
from datetime import datetime
from scipy.signal import correlate

from q_gen import get_rho_q, get_q_points_all_quads, get_binning_averages
from flames import get_static_sf, order_q_points, get_ISF_corr

# -----------------------------------------------------------------------------

if __name__ == '__main__':

	# input parameters
	q_end = 1.0
	max_points = 2000
	num_q_bins = 50
	Nthreads = 64

	trjfile = "../data/sampling_100frames_100ns.xtc"
	grofile = "../data/md_0_300.gro"
	Fr_start=0
	Fr_step=1
	Fr_stop=2000
	Fr_total=2e5
	dt_traj=0.5 # ps

    # use adaptive sampling or not: True/False
	q_adaptive_bool = False

    # ---------------------------------------------------------------------direct method
	# print information	
	numba.set_num_threads(Nthreads)  # Set the number of threads
	print(f"\nComplex fluids analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {numba.get_num_threads()} threads\n")

	# load the trajectory files of Gromacs	
	u = mda.Universe(grofile, trjfile)
	system = u.select_atoms('all')
	atomnames = u.atoms.names
	formfact_all = np.array([33 if i<len(atomnames)*0.5 else 0 for i in range(len(atomnames))])

	# get box info
	bx, by, bz = u.dimensions[:3]

	# get q-points	
	q_points = get_q_points_all_quads(np.array([bx, by, bz]), q_end, max_points=max_points)
	np.save(f"q_points.npy", q_points)
	
	if q_adaptive_bool:

		#static structure factor
		ssf = get_static_sf(q_points, system, u.trajectory[Fr_start:Fr_stop+1:Fr_step], formfact_all)
		np.save(f"s_q_t.npy", ssf)

		# binning averages
		qr, ssf_qr = get_binning_averages(num_q_bins, q_end, ssf, q_points)
		np.save(f"qrlist_ssf.npy", qr)
		np.save(f"s_q_t_binned.npy", ssf_qr)

		print(f"\nStatic structure factor calculation done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

		# order it
		q_points_binned = order_q_points(q_points, q_end)
		dt_in_ps = np.array([50, 15, 2.5, 0.5]) # in ps

		# test 2000 frames for all for the accuracy
		Nfrs_used = 2000
		Frs_sep = (dt_in_ps/dt_traj).astype(int) # array([100,  30,  5,   1])

		for iq_idx in range(len(q_points_binned)):
			i_qpts = q_points_binned[iq_idx]
			np.save(f"q_points_{iq_idx}.npy", i_qpts)

			Fr_start=0
			Fr_step = Frs_sep[iq_idx]
			Fr_stop = min(Nfrs_used*Fr_step, int(Fr_total)) # trim with the max frames

			# get ISF
			isf = get_ISF_corr(i_qpts, system, u.trajectory[Fr_start:Fr_stop+1:Fr_step], formfact_all)
			np.save(f"F_q_t_{iq_idx}.npy", isf)

			Fr_list = np.array(list(range(Fr_start,Fr_stop+1,Fr_step)))
			np.save(f"Fr_list_{iq_idx}.npy", Fr_list)

			# get g1/2
			g1 = np.zeros(isf.shape)
			g2 = np.zeros(isf.shape)
			for idx in range(isf.shape[0]):
				g1[idx, :] = isf[idx,:]/isf[idx,0]
				g2[idx, :] = 1 + (isf[idx,:]/isf[idx,0])**2

			np.save(f"g1_{iq_idx}.npy", g1)
			np.save(f"g2_{iq_idx}.npy", g2)

			print(f"\nISF and g2 calculation done for {iq_idx}-th qpoints at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

	else:
		# get static sf
		ssf = get_static_sf(q_points, system, u.trajectory[Fr_start:Fr_stop+1:Fr_step], formfact_all)
		np.save(f"s_q_t.npy", ssf)

		# binning averages		
		qr, ssf_qr = get_binning_averages(num_q_bins, q_end, ssf, q_points)
		np.save(f"qrlist_ssf.npy", qr)
		np.save(f"s_q_t_binned.npy", ssf_qr)

		print(f"\nStatic structure factor calculation done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")			
			
		# get ISF
		isf = get_ISF_corr(q_points, system, u.trajectory[Fr_start:Fr_stop+1:Fr_step], formfact_all)
		np.save(f"F_q_t.npy", isf)
		Fr_list = np.array(list(range(Fr_start,Fr_stop+1,Fr_step)))
		np.save(f"Fr_list.npy", Fr_list)

		# binning averages
		qr, dsf_qr = get_binning_averages(num_q_bins, q_end, isf, q_points)
		np.save(f"qrlist_isf.npy", qr)
		np.save(f"F_q_t_binned.npy", dsf_qr)

		print(f"\nISF calculation done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

		# get g1/2
		g1 = np.zeros(isf.shape)		
		g2 = np.zeros(isf.shape)
		for idx in range(isf.shape[0]):
			g1[idx, :] = isf[idx,:]/isf[idx,0]
			g2[idx, :] = 1 + (isf[idx,:]/isf[idx,0])**2

		np.save(f"g1.npy", g1)
		np.save(f"g2.npy", g2)

		# binning averages
		qr_g1, g1_qr = get_binning_averages(num_q_bins, q_end, g1, q_points)
		qr_g2, g2_qr = get_binning_averages(num_q_bins, q_end, g2, q_points)
		np.save(f"qrlist_g1.npy", qr_g1)
		np.save(f"qrlist_g2.npy", qr_g2)
		np.save(f"g1_binned.npy", g1_qr)
		np.save(f"g2_binned.npy", g2_qr)

		print(f"\ng2 from ISF calculation using Siegert relation done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
