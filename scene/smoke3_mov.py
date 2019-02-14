import argparse
from datetime import datetime
import os
from tqdm import trange
import numpy as np
from PIL import Image
import gc
try:
	from manta import *
except ImportError:
	pass

from collections import deque
from perlin import TileableNoise

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke3_mov200_f400')
parser.add_argument("--num_param", type=int, default=2)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--p0", type=str, default='scenes')
parser.add_argument("--p1", type=str, default='frames')

num_s = 200
num_f = 400
num_sim = num_s*num_f
parser.add_argument("--min_src_pos", type=float, default=0.1)
parser.add_argument("--max_src_pos", type=float, default=0.9)
parser.add_argument("--src_y_pos", type=float, default=0.1)
parser.add_argument("--src_radius", type=float, default=0.08)
parser.add_argument("--min_scenes", type=int, default=0)
parser.add_argument("--max_scenes", type=int, default=num_s-1)
parser.add_argument("--num_scenes", type=int, default=num_s)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_f-1)
parser.add_argument("--num_frames", type=int, default=num_f)
parser.add_argument("--num_simulations", type=int, default=num_sim)
parser.add_argument("--num_dof", type=int, default=2)

parser.add_argument("--resolution_x", type=int, default=48) # 96
parser.add_argument("--resolution_y", type=int, default=72) # 128
parser.add_argument("--resolution_z", type=int, default=48) # 1
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=str, default='xXyYzZ') # xXyY
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--adv_order", type=int, default=2)
parser.add_argument("--clamp_mode", type=int, default=2)

parser.add_argument("--nscale", type=float, default=0.01)
parser.add_argument("--nrepeat", type=int, default=1000)
parser.add_argument("--nseed", type=int, default=123)

args = parser.parse_args()

def advect():
	def get_param(p1):
		min_p1 = args.min_scenes
		max_p1 = args.max_scenes
		num_p1 = args.num_scenes
		p1_ = p1/(num_p1-1) * (max_p1-min_p1) + min_p1
		return p1_

	p1 = 0
	# p1_ = get_param(p1)
	v_path = os.path.join(args.log_dir, 'v')
	img_dir = os.path.join(args.log_dir, 'd_adv')
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z
	gs = vec3(res_x, res_y, res_z)

	dim = 2
	if res_z > 1: dim = 3
	s = Solver(name='main', gridSize=gs, dim=dim)
	s.timestep = args.time_step

	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)

	flags.initDomain(boundaryWidth=args.bWidth)
	flags.fillGrid()
	setOpenBound(flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty)

	vel.clear()
	density.clear()

	radius = gs.x*args.src_radius

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		# gui.pause()

	if res_z > 1:
		d_ = np.zeros([res_z, res_y, res_x], dtype=np.float32)
	else:
		d_ = np.zeros([res_y, res_x], dtype=np.float32)
	for t in trange(args.num_frames):
		v_path_ = os.path.join(v_path, args.path_format % (p1, t))
		with np.load(v_path_) as data:
			v = data['x']
			if res_z == 1:
				v = np.dstack((v,np.zeros([res_y, res_x, 1])))
			p = data['y']

		copyArrayToGridMAC(v, vel)

		nx = p[0,-1]
		nz = 0.5
		if res_z > 1: nz = p[1,-1]
		source = s.create(Sphere, center=gs*vec3(nx,args.src_y_pos,nz), radius=radius)
		source.applyToGrid(grid=density, value=1)			
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
							openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
		
		copyGridToArrayReal(density, d_)

		img_path = os.path.join(img_dir, '%04d.png' % t)
		if res_z > 1:
			d_img = np.mean(d_[:,::-1], axis=0)*255
		else:
			d_img = d_[::-1]*255
		d_img = np.stack((d_img,d_img,d_img), axis=-1).astype(np.uint8)
		d_img = Image.fromarray(d_img)
		d_img.save(img_path)
		s.step()

def nplot():
	import matplotlib.pyplot as plt

	n_path = os.path.join(args.log_dir, 'n.npz')
	nz_list = None
	with np.load(n_path) as data:		
		nx_list = data['nx']
		if 'nz' in data:
			nz_list = data['nz']

	t = range(args.num_frames)
	fig = plt.figure()

	if nz_list is None:
		plt.ylim((0,1))
		for i in range(args.num_scenes):
			plt.plot(t, nx_list[i,:])
	else:
		plt.subplot(211)
		plt.ylim((0,1))
		for i in range(args.num_scenes):
			plt.plot(t, nx_list[i,:])

		plt.subplot(212)
		plt.ylim((0,1))
		for i in range(args.num_scenes):
			plt.plot(t, nz_list[i,:])

	n_fig_path = os.path.join(args.log_dir, 'n.png')
	fig.savefig(n_fig_path)	

def main():
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	field_type = ['v'] #, 'p', 'd'
	for field in field_type:
		field_path = os.path.join(args.log_dir,field)
		if not os.path.exists(field_path):
			os.makedirs(field_path)

	args_file = os.path.join(args.log_dir, 'args.txt')
	with open(args_file, 'w') as f:
		print('%s: arguments' % datetime.now())
		for k, v in vars(args).items():
			print('  %s: %s' % (k, v))
			f.write('%s: %s\n' % (k, v))

	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z

	if res_z > 1:
		v_ = np.zeros([res_z,res_y,res_x,3], dtype=np.float32)
		# p_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
		# d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
		# s_ = np.zeros([res_z,res_y,res_x,3], dtype=np.float32)
	else:
		v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
		# p_ = np.zeros([res_y,res_x], dtype=np.float32)
		# d_ = np.zeros([res_y,res_x], dtype=np.float32)
		# s_ = np.zeros([res_y,res_x], dtype=np.float32)

	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]	

	noise = TileableNoise(seed=args.nseed)


	# solver params
	gs = vec3(res_x, res_y, res_z)
	dim = 2
	if res_z > 1: dim = 3
	s = Solver(name='main', gridSize=gs, dim=dim)
	s.timestep = args.time_step

	buoyancy = vec3(0,args.buoyancy,0)
	radius = gs.x*args.src_radius
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)
	pressure = s.create(RealGrid)
	# stream = s.create(RealGrid)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		#gui.pause()

	print('start generation')
	nx_list = []
	if res_z > 1: nz_list = []

	for i in trange(args.num_scenes, desc='scenes'):
		flags.initDomain(boundaryWidth=args.bWidth)
		flags.fillGrid()
		setOpenBound(flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty)

		vel.clear()
		density.clear()
		pressure.clear()
		# stream.clear()
		
		# noise
		noise.randomize()
		ny_ = noise.rng.randint(200)*args.nscale
		nz_ = noise.rng.randint(200)*args.nscale
		nqx = deque([-1]*args.num_frames,args.num_frames)
		if res_z > 1:
			nx_ = noise.rng.randint(200)*args.nscale
			nqz = deque([-1]*args.num_frames,args.num_frames)
		
		for t in trange(args.num_frames, desc='sim', leave=False):
			nx = noise.noise3(x=t*args.nscale, y=ny_, z=nz_, repeat=args.nrepeat)
			px = (nx+1)*0.5 * (args.max_src_pos-args.min_src_pos) + args.min_src_pos # [minx, maxx]
			pz = 0.5
			nqx.append(px)
			param_ = [list(nqx)]
			if res_z > 1:
				nz = noise.noise3(x=nx_, y=ny_, z=t*args.nscale, repeat=args.nrepeat)
				pz = (nz+1)*0.5 * (args.max_src_pos-args.min_src_pos) + args.min_src_pos # [minx, maxx]
				nqz.append(pz)
				param_ = [list(nqx), list(nqz)]
			param_ = np.array(param_)

			source = s.create(Sphere, center=gs*vec3(px,args.src_y_pos,pz), radius=radius)
			source.applyToGrid(grid=density, value=1)
				
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
							   openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=args.adv_order,
							   openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			setWallBcs(flags=flags, vel=vel)
			addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
			solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
			setWallBcs(flags=flags, vel=vel)
			# getStreamfunction(flags=flags, vel=vel, grid=stream)
		
			copyGridToArrayMAC(vel, v_)
			# copyGridToArrayReal(pressure, p_)
			# copyGridToArrayReal(density, d_)
			# copyGridToArrayReal(stream, s_)
			
			v_range = [np.minimum(v_range[0], v_.min()),
					   np.maximum(v_range[1], v_.max())]
			# p_range = [np.minimum(p_range[0], p_.min()),
			# 		   np.maximum(p_range[1], p_.max())]
			# d_range = [np.minimum(d_range[0], d_.min()),
			# 		   np.maximum(d_range[1], d_.max())]
			# s_range = [np.minimum(s_range[0], s_.min()),
			# 		   np.maximum(s_range[1], s_.max())]

			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, t))
			if res_z > 1:
				np.savez_compressed(v_file_path,
									x=v_,
									y=param_)
			else:
				np.savez_compressed(v_file_path,
									x=v_[...,:2],
									y=param_)

			# p_file_path = os.path.join(args.log_dir, 'p', args.path_format % (i, t))
			# np.savez_compressed(p_file_path,
			# 					x=p_,
			# 					y=param_)

			# d_file_path = os.path.join(args.log_dir, 'd', args.path_format % (i, t))
			# np.savez_compressed(d_file_path,
			# 					x=d_,
			# 					y=param_)

			s.step()

		nx_list.append(list(nqx))
		if res_z > 1:
			nz_list.append(list(nqz))
		gc.collect()

	n_path = os.path.join(args.log_dir, 'n.npz')
	if res_z > 1:
		np.savez_compressed(n_path, nx=nx_list, nz=nz_list)
	else:
		np.savez_compressed(n_path, nx=nx_list)

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	# prange_file = os.path.join(args.log_dir, 'p_range.txt')
	# with open(prange_file, 'w') as f:
	# 	print('%s: pressure min %.3f max %.3f' % (datetime.now(), p_range[0], p_range[1]))
	# 	f.write('%.3f\n' % p_range[0])
	# 	f.write('%.3f' % p_range[1])

	# drange_file = os.path.join(args.log_dir, 'd_range.txt')
	# with open(drange_file, 'w') as f:
	# 	print('%s: density min %.3f max %.3f' % (datetime.now(), d_range[0], d_range[1]))
	# 	f.write('%.3f\n' % d_range[0])
	# 	f.write('%.3f' % d_range[1])

	# srange_file = os.path.join(args.log_dir, 's_range.txt')
	# with open(srange_file, 'w') as f:
	# 	print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
	# 	f.write('%.3f\n' % s_range[0])
	# 	f.write('%.3f' % s_range[1])

	print('Done')

if __name__ == '__main__':
	main()

	# source trajectory
	nplot()

	# advection test
	advect()