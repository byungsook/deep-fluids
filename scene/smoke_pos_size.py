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

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke_pos21_size5_f200')
parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='src_x_pos')
parser.add_argument("--p1", type=str, default='src_radius')
parser.add_argument("--p2", type=str, default='frames')

parser.add_argument("--num_src_x_pos", type=int, default=21)
parser.add_argument("--min_src_x_pos", type=float, default=0.2)
parser.add_argument("--max_src_x_pos", type=float, default=0.8)
parser.add_argument("--src_y_pos", type=float, default=0.1)
parser.add_argument("--num_src_radius", type=int, default=5)
parser.add_argument("--min_src_radius", type=float, default=0.04)
parser.add_argument("--max_src_radius", type=float, default=0.12)
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=199)
parser.add_argument("--num_simulations", type=int, default=21000)

parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=128)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--adv_order", type=int, default=2)
parser.add_argument("--clamp_mode", type=int, default=2)

args = parser.parse_args()

def advect():
	def get_param(p1, p2):
		min_p1 = args.min_src_x_pos
		max_p1 = args.max_src_x_pos
		num_p1 = args.num_src_x_pos
		min_p2 = args.min_src_radius
		max_p2 = args.max_src_radius
		num_p2 = args.num_src_radius
		p1_ = p1/(num_p1-1) * (max_p1-min_p1) + min_p1
		p2_ = p2/(num_p2-1) * (max_p2-min_p2) + min_p2
		return p1_, p2_

	p1, p2 = 10, 2
	p1_, p2_ = get_param(p1, p2)
	v_path = os.path.join(args.log_dir, 'v')
	img_dir = os.path.join(args.log_dir, 'd_adv')
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	gs = vec3(res_x, res_y, 1)

	s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)

	flags.initDomain(boundaryWidth=args.bWidth)
	flags.fillGrid()
	if args.open_bound:
		setOpenBound(flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

	source = s.create(Sphere, center=gs*vec3(p1_,args.src_y_pos,0.5), radius=gs.x*p2_)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.pause()

	d_ = np.zeros([res_y, res_x], dtype=np.float32)
	for t in trange(args.num_frames):
		v_path_ = os.path.join(v_path, args.path_format % (p1, p2, t))
		with np.load(v_path_) as data:
			v = data['x']
			v = np.dstack((v,np.zeros([res_y, res_x, 1])))

		copyArrayToGridMAC(v, vel)
		source.applyToGrid(grid=density, value=1)			
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
							openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
		copyGridToArrayReal(density, d_)

		img_path = os.path.join(img_dir, '%04d.png' % t)
		d_img = d_[::-1]*255
		d_img = np.stack((d_img,d_img,d_img), axis=-1).astype(np.uint8)
		d_img = Image.fromarray(d_img)
		d_img.save(img_path)
		s.step()

def main():
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	field_type = ['v'] #, 'd']
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

	p1_space = np.linspace(args.min_src_x_pos, 
						   args.max_src_x_pos,
						   args.num_src_x_pos)
	p2_space = np.linspace(args.min_src_radius,
						   args.max_src_radius,
						   args.num_src_radius)
	p_list = np.array(np.meshgrid(p1_space, p2_space)).T.reshape(-1, 2)
	pi1_space = range(args.num_src_x_pos)
	pi2_space = range(args.num_src_radius)
	pi_list = np.array(np.meshgrid(pi1_space, pi2_space)).T.reshape(-1, 2)

	res_x = args.resolution_x
	res_y = args.resolution_y
	v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
	# d_ = np.zeros([res_y,res_x], dtype=np.float32)
	# p_ = np.zeros([res_y,res_x], dtype=np.float32)
	# s_ = np.zeros([res_y,res_x], dtype=np.float32)

	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# d_range = [np.finfo(np.float).max, np.finfo(np.float).min] # 0-1
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]


	# solver params
	gs = vec3(res_x, res_y, 1)
	buoyancy = vec3(0, args.buoyancy, 0)

	s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)
	pressure = s.create(RealGrid)
	stream = s.create(RealGrid)

	if (GUI):
		gui = Gui()
		gui.show(True)
		#gui.pause()

	print('start generation')
	for i in trange(len(p_list), desc='scenes'):
		flags.initDomain(boundaryWidth=args.bWidth)
		flags.fillGrid()
		if args.open_bound:
			setOpenBound(flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

		vel.clear()
		density.clear()
		pressure.clear()
		stream.clear()
		
		p0, p1 = p_list[i][0], p_list[i][1]
		radius = gs.x*p1
		source = s.create(Sphere, center=gs*vec3(p0,args.src_y_pos,0.5), radius=radius)
		
		for t in trange(args.num_frames, desc='sim'):
			source.applyToGrid(grid=density, value=1)
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
							   openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=args.adv_order,
							   openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			setWallBcs(flags=flags, vel=vel)
			addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
			solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
			setWallBcs(flags=flags, vel=vel)
			# getStreamfunction(flags=flags, vel=vel, grid=stream)
		
			copyGridToArrayMAC(vel, v_)
			# copyGridToArrayReal(density, d_)
			# copyGridToArrayReal(pressure, p_)
			# copyGridToArrayReal(stream, s_)
			
			v_range = [np.minimum(v_range[0], v_.min()),
					   np.maximum(v_range[1], v_.max())]
			# d_range = [np.minimum(d_range[0], d_.min()),
			# 		   np.maximum(d_range[1], d_.max())]
			# p_range = [np.minimum(p_range[0], p_.min()),
			# 		   np.maximum(p_range[1], p_.max())]
			# s_range = [np.minimum(s_range[0], s_.min()),
			# 		   np.maximum(s_range[1], s_.max())]

			param_ = [p0, p1, t]
			pit = tuple(pi_list[i].tolist() + [t])

			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
			np.savez_compressed(v_file_path, 
								x=v_[...,:2],
								y=param_)

			# d_file_path = os.path.join(args.log_dir, 'd', args.path_format % pit)[:-3] + 'png'
			# d_img = d_[::-1,:]*255
			# d_img = np.stack((d_img,d_img,d_img), axis=-1).astype(np.uint8)
			# d_img = Image.fromarray(d_img)
			# d_img.save(d_file_path)

			s.step()
		
		gc.collect()

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	# drange_file = os.path.join(args.log_dir, 'd_range.txt')
	# with open(drange_file, 'w') as f:
	# 	print('%s: density min %.3f max %.3f' % (datetime.now(), d_range[0], d_range[1]))
	# 	f.write('%.3f\n' % d_range[0])
	# 	f.write('%.3f' % d_range[1])

	# prange_file = os.path.join(args.log_dir, 'p_range.txt')
	# with open(prange_file, 'w') as f:
	# 	print('%s: pressure min %.3f max %.3f' % (datetime.now(), p_range[0], p_range[1]))
	# 	f.write('%.3f\n' % p_range[0])
	# 	f.write('%.3f' % p_range[1])

	# srange_file = os.path.join(args.log_dir, 's_range.txt')
	# with open(srange_file, 'w') as f:
	# 	print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
	# 	f.write('%.3f\n' % s_range[0])
	# 	f.write('%.3f' % s_range[1])

	print('Done')


if __name__ == '__main__':
	main()

	# advection test
	advect()