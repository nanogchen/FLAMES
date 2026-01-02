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

import itertools
import numpy as np
from numba import njit, prange

def get_q_points_slab(box, q_max, normal='y'):
	"""
	construct q-points in a slab: xz with norm-y
	or xy with norm z
	"""

	if normal == 'y':
		idx_list = [0,2]

	elif normal == 'z':
		idx_list = [0,1]

	elif normal == 'x':
		idx_list = [1,2]

	else:
		print(f"Not allowed value of normal ({normal})! Can ONLY be x/y/z")
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

	return q_points

"""from dynasor, with some modification"""

@njit(fastmath=True, nogil=True)
def dot3(a,b):
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(fastmath=True, nogil=True, parallel=True)
def get_rho_q(x, q, formfact_all):

	"""
	get rho of q over all q-points with given form factor
	"""

	Nx = len(x)
	Nq = len(q)

	rho_q = np.zeros(Nq, dtype=np.complex128)

	# parallel
	for iq in prange(Nq):

		rho = 0.0j
		for ix in range(Nx):
			alpha = dot3(x[ix], q[iq])
			rho += np.exp(-1j * alpha) * formfact_all[ix]

		rho_q[iq] = rho

	return rho_q

@njit(fastmath=True, nogil=True, parallel=True)
def get_rho_q_noFF(x, q):

	"""
	get rho of q over all q-points w/o form factor
	"""

	Nx = len(x)
	Nq = len(q)

	rho_q = np.zeros(Nq, dtype=np.complex128)

	# prange is like OMP
	for iq in prange(Nq):

		rho = 0.0j
		for ix in range(Nx):
			alpha = dot3(x[ix], q[iq])
			rho += np.exp(-1j * alpha)

		rho_q[iq] = rho

	return rho_q

def get_prune_distance(max_points, q_max, q_vol):
	"""from dynasor: originally just first-quadrant"""
	Q = q_max
	V = q_vol
	N = max_points

	# Coefs
	a = 1.0
	b = -3 / 2 * Q
	c = 0.0
	d = 3 / 2 * V * N / (4 * np.pi)

	# Eq tol solve
	def original_eq(x):
		return a * x**3 + b * x**2 + c * x + d
	# original_eq = lambda x:  a * x**3 + b * x**2 + c * x + d

	# Discriminant
	p = (3 * a * c - b**2) / (3 * a**2)
	q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

	D_t = - (4 * p**3 + 27 * q**2)
	if D_t < 0:
		return q_max

	x = Q * (np.cos(1 / 3 * np.arccos(1 - 4 * d / Q**3) - 2 * np.pi / 3) + 0.5)

	assert np.isclose(original_eq(x), 0, rtol=1e-05, atol=1e-06), original_eq(x)

	return x

def get_q_points_all_quads(box, q_max, max_points, seed=42):

	"""
	get q vectors/points: mod from dynasor (first-quad)

	input:
	box: np.array([lx, ly, lz])
	q_max: float max q
	"""

	dq = np.diagflat(2*np.pi/box)
	N = np.ceil(q_max/np.diag(dq)).astype(int)

	lattice_points = list(itertools.product(*[range(-n, n+1) for n in N]))
	q_points = lattice_points @ dq

	# distance measure and sort out
	q_dist = np.linalg.norm(q_points, axis=1)
	argsort = np.argsort(q_dist)
	q_dist = q_dist[argsort]
	q_points = q_points[argsort]

	# prune based on max_q
	q_points = q_points[q_dist <= q_max]
	q_dist = q_dist[q_dist <= q_max]

	# prune based on max_Npoints
	if max_points < len(q_points):

		q_vol = dq[0,0]*dq[1,1]*dq[2,2]
		q_prune = get_prune_distance(max_points, q_max, q_vol)

		if q_prune < q_max:
			print(f'Pruning q-points from the range {q_prune:.3} < |q| < {q_max}')
			p = np.ones(len(q_points))
			assert np.isclose(q_dist[0], 0)
			p[1:] = (q_prune / q_dist[1:]) ** 2

			rs = np.random.RandomState(seed)
			q_points = q_points[p > rs.rand(len(q_points))]
			print(f'Pruned from {len(q_dist)} q-points to {len(q_points)}')

	return q_points

def get_binning_averages(num_q_bins, q_end, data_in_q_t, q_points):
	""" get function of q_norm by binning"""

	# do binning
	Nframes = data_in_q_t.shape[1]
	q_norms = np.linalg.norm(q_points, axis=1)

	# setup bins
	bin_counts, edges = np.histogram(q_norms, bins=num_q_bins, range=(0.0, q_end))
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

