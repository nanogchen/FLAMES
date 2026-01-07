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
from numba import njit, prange
from datetime import datetime
from scipy.signal import correlate

import sys
sys.path.insert(0, "../")
from flames.q_gen import get_rho_q,get_rho_q_noFF,get_q_points_all_quads,get_binning_averages,get_binning_averages_ttc
from flames.q_gen import get_q_points_plane,get_q_points_angular_bin

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

	q1,q2,q_points = get_q_points_plane(box, q_max, plane)

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

def get_ttc(box, q_min, q_max, Nbins, angle_deg, system, traj, formfact_all, plane='xz'):
	"""
	get two-time correlation C(q,t1,t2).
	"""		

	# generate q-points
	q_points = get_q_points_angular_bin(box, q_min, q_max, Nbins, angle_deg, plane)

	# first get s(q,t)
	n_qpoints = len(q_points)
	n_frs = len(traj)
	ssf = np.zeros((n_qpoints, n_frs))

	# get S(q,t)
	ifr=0    
	for _ in traj:

		coords = system.positions
		rho_q = get_rho_q(coords, q_points, formfact_all)
		sq_t = np.real(rho_q*rho_q.conjugate()) 

		ssf[:,ifr] = sq_t
		ifr+=1

	# get C(q, t1, t2) = <I1*I2>
	I_q_t1_t2 = np.zeros((n_qpoints, n_frs, n_frs))
	for iq in prange(n_qpoints):
		I_q_t1_t2[iq] = np.outer(ssf[iq],ssf[iq])
	
	return q_points, ssf, I_q_t1_t2

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

def symm_func(t, inverse=True):
	"""symmetric a sequence"""

	if inverse:
		t_ = t[::-1]*(-1)
	else:
		t_ = t[::-1]
	t_ = list(t_[:-1])

	t_.extend(list(t))

	return np.array(t_)

def fourier_transform_1d(x, fx):
	x0, dx = x[0], x[1] - x[0]
	g = np.fft.fft(fx) # DFT calculation
	
	# frequency normalization factor is 2*np.pi/dt
	w = np.fft.fftfreq(x.size)*2*np.pi/dx # angular frequency
	
	# Multiply by external factor
	g *= dx*np.exp(-complex(0,1)*w*x0) 
	
	return w,g

def fft_dft_symm(Tseq, fTseq):
	"""do Fourier tranformation on the autocorrelation functions, e.g, vel-acf using self-code
	This gives the same results as the Filon formula
	"""

	# symmetrize the input
	T_symm = symm_func(Tseq)
	fT_symm = symm_func(fTseq, inverse=False)

	# do FT
	w, cw = fourier_transform_1d(T_symm, fT_symm)

	return w[:len(w)//2], np.abs(cw)[:len(w)//2]
	# return w, cw