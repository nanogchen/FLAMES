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

import sys
sys.path.insert(0, "/home/softmatter/flames/flames")
from q_gen import get_rho_q, get_rho_q_noFF, get_q_points_all_quads, get_binning_averages

def get_static_sf(q_points, system, traj, formfact_all):

	"""
	get static structure factor S(q_vec, t) at given time.
	"""
	n_qpoints = len(q_points)
	ssf = np.zeros((n_qpoints, len(traj)))

	ifr=0    
	for _ in traj:

		coords = system.positions
		rho_q = get_rho_q(coords, q_points, formfact_all)
		sq_t = np.real(rho_q*rho_q.conjugate()) 

		ssf[:,ifr] = sq_t
		ifr+=1

	return ssf/(np.sum(formfact_all**2))

def get_sf_decomposition(q_points, ag1, ag2, traj):
	"""
	get decomposition of the static structure factor
	note: no manual-set form factor is used here!
	"""

	n_qpoints = len(q_points)
	sf_AA = np.zeros((n_qpoints, len(traj)))
	sf_AB = np.zeros((n_qpoints, len(traj)))
	sf_BB = np.zeros((n_qpoints, len(traj)))    

	ifr=0
	for _ in traj:

		coords_A = ag1.positions
		coords_B = ag2.positions

		rho_qA = get_rho_q_noFF(coords_A, q_points)
		rho_qB = get_rho_q_noFF(coords_B, q_points)

		sf_AA[:,ifr] = np.real(rho_qA*rho_qA.conjugate())
		sf_BB[:,ifr] = np.real(rho_qB*rho_qB.conjugate())
		sf_AB[:,ifr] = np.real(rho_qA*rho_qB.conjugate())+np.real(rho_qB*rho_qA.conjugate()) # considered two

		ifr+=1

	Natoms = ag1.atoms.n_atoms + ag2.atoms.n_atoms
	return sf_AA/Natoms, 0.5*sf_AB/Natoms, sf_BB/Natoms

def get_scattering_image(box, q_max, system, traj, plane='xz'):
	"""
	construct q-points in a plane
	"""

	if plane == 'xz':
		idx_list = [0,2]
		xlabel = 'x'
		ylabel = 'z'

	elif plane == 'xy':
		idx_list = [0,1]
		xlabel = 'x'
		ylabel = 'y'

	elif plane == 'yz':
		# just for completeness, may not be used
		idx_list = [1,2]
		xlabel = 'y'
		ylabel = 'z'

	else:
		print(f"Not allowed value of plane ({plane})! Can ONLY be xy/yz/xz")
		sys.exit(0)

	box2 = box[idx_list]
	dq = np.diagflat(2*np.pi/box2)
	N = np.ceil(q_max/np.diag(dq)).astype(int)

	# form the q-points in each of the two direction
	q1 = np.arange(-N[0], N[0]+1) * dq[0,0] # 2*N+1
	q2 = np.arange(-N[1], N[1]+1) * dq[1,1]

	# the q-points
	qpts1, qpts2 = np.meshgrid(q1, q2)
	q_points2 = np.stack((qpts1.ravel(), qpts2.ravel()), axis=-1)
	q_points = np.zeros((q_points2.shape[0], 3))
	q_points[:, idx_list] = q_points2

	# out array
	ssf_1d = np.zeros((len(q_points), len(traj)))
	ssf_2d = np.zeros((q1.shape[0], q2.shape[0], len(traj)))

	ifr=0    
	for _ in traj:

		coords = system.positions
	
		# cal sf. at each q-points
		rho_q = get_rho_q_noFF(coords, q_points)
		ssf = np.real(rho_q*rho_q.conjugate()) / coords.shape[0] # 1/N
		ssf_1d[:, ifr] = ssf
		ssf_2d[:, :, ifr] = np.reshape(ssf, (q1.shape[0], q2.shape[0]))

	return q_points, ssf_1d, q1, q2, ssf_2d

def get_ISF_corr(q_points, system, traj, formfact_all):

	"""
	get the ISF using autocorrlation function of density field
	"""

	n_qpoints = len(q_points)
	rho_qt = np.zeros(shape=(n_qpoints, len(traj)), dtype=np.complex128)

	# get rho(q,t)
	ifr = 0
	for _ in traj:

		coords = system.positions
		rho_q = get_rho_q(coords, q_points, formfact_all) 
		rho_qt[:, ifr] = rho_q
		ifr += 1

	# do autocorrelation
	isf = np.zeros((n_qpoints, len(traj)))
	for iq in range(n_qpoints):
		rho_qi = rho_qt[iq, :]
		acf_rho_full = correlate(rho_qi,rho_qi,mode='full')
		acf_rho = acf_rho_full[len(acf_rho_full)//2:]
		acf_rho_ave = np.divide(acf_rho, np.linspace(len(acf_rho), 1, num=len(acf_rho), endpoint=True))

		isf[iq, :] = np.real(acf_rho_ave)

	return isf/(np.sum(formfact_all**2))

def order_q_points(q_points, q_max):
	"""
	order q by norm
	"""

	factor = math.sqrt(10)
	q_min = 0.02
	q_hi = q_min
	
	q_bin = []
	q_bin.append(q_hi)
	while q_hi <= q_max:
		q_hi *= factor
		q_bin.append(q_hi)

	# divide into bins: find the indices of bin
	q_norm = np.linalg.norm(q_points, axis=1)
	indices = np.searchsorted(q_bin, q_norm, side='right')

	# put into bins
	q_points_binned = []
	for ibin in range(len(q_bin)-1): # ignore q=[0,0,0]
		q_pts_ibin = []
		for iq in range(1,len(q_points)):
			idx = indices[iq]-1
			if idx == ibin:
				q_pts_ibin.append(q_points[iq])
			
		q_points_binned.append(np.array(q_pts_ibin))
	
	return q_points_binned

def binning_local(data_in_q_t, q_points):
	""" get function of q_norm by binning for selective q-range"""

	# do binning
	Nframes = data_in_q_t.shape[1]
	q_norms = np.linalg.norm(q_points, axis=1)

	# setup bins
	bin_size = 0.02
	q_max = np.max(q_norms)
	q_min = np.min(q_norms)
	num_q_bins = math.ceil((q_max - q_min)/bin_size)
	dqr = (q_max - q_min) / (num_q_bins - 1)
	q_range = (q_min - dqr / 2, q_max + dqr / 2)
	bin_counts, edges = np.histogram(q_norms, bins=num_q_bins, range=q_range)
	q_bincenters = 0.5 * (edges[1:] + edges[:-1])

	# calculate average for each bin
	averaged_data = np.zeros((num_q_bins, Nframes))
	for bin_index in range(num_q_bins):
		# find q-indices that belong to this bin
		bin_min = edges[bin_index]
		bin_max = edges[bin_index + 1]
		bin_count = bin_counts[bin_index]
		q_indices = np.where(np.logical_and(q_norms >= bin_min, q_norms < bin_max))[0]

		# average over q-indices, if no indices then np.nan
		if bin_count == 0:
			print(f'No q-points for bin {bin_index}')
			data_bin = np.array([np.nan for _ in range(Nframes)])
		else:
			data_bin = data_in_q_t[q_indices, :].mean(axis=0)
		averaged_data[bin_index, :] = data_bin

	return q_bincenters, averaged_data

# -----------------------------------------------------------------------------

if __name__ == '__main__':

	# input parameters
	q_end = 1.0
	max_points = 2000
	num_q_bins = 50
	Nthreads = 64

	trjfile = "sampling_100ns.trr"
	grofile = "md_0_300.gro"
	Fr_start=0
	Fr_step=1
	Fr_stop=2000
	Fr_total=2e5
	dt_traj=0.5 # ps

    # use adaptive sampling or not: True/False
	q_adaptive_bool = True

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
