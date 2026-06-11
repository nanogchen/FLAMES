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
from flames.q_gen import get_rho_q,get_rho_q_noFF,get_q_points_all_quads,get_binning_averages

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
