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

parser.add_argument("--log_dir", type=str, default='data/liquid_pos10_size4_f200')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='src_x_pos')
parser.add_argument("--p1", type=str, default='src_radius')
parser.add_argument("--p2", type=str, default='frames')

parser.add_argument("--num_src_x_pos", type=int, default=10)
parser.add_argument("--min_src_x_pos", type=float, default=0.2)
parser.add_argument("--max_src_x_pos", type=float, default=0.8)
parser.add_argument("--src_y_pos", type=float, default=0.6)
parser.add_argument("--num_src_radius", type=int, default=4)
parser.add_argument("--min_src_radius", type=float, default=0.04)
parser.add_argument("--max_src_radius", type=float, default=0.08)
parser.add_argument("--basin_y_pos", type=float, default=0.2)
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=199)
parser.add_argument("--num_simulations", type=int, default=8000)

parser.add_argument("--resolution_x", type=int, default=128)
parser.add_argument("--resolution_y", type=int, default=64)
parser.add_argument("--gravity", type=float, default=-1e-3)
parser.add_argument("--radius_factor", type=float, default=1)
parser.add_argument("--min_particles", type=int, default=2)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.5)

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

	# p1, p2 = 4, 1
	p1, p2 = 0, 0
	p1_, p2_ = get_param(p1, p2)
	v_path = os.path.join(args.log_dir, 'v')
	img_dir = os.path.join(args.log_dir, 'l_adv')
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

	pp = s.create(BasicParticleSystem) 

	# acceleration data for particle nbs
	pindex = s.create(ParticleIndexSystem) 
	gpi = s.create(IntGrid)

	flags.initDomain(boundaryWidth=args.bWidth)
	vel.clear()
	pp.clear()
	pindex.clear()
	gpi.clear()

	fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
	dropCenter = vec3(p1_,args.src_y_pos,0.5)
	dropRadius = p2_
	fluidDrop = Sphere(parent=s, center=gs*dropCenter, radius=gs.x*dropRadius)
	phi = fluidBasin.computeLevelset()
	phi.join(fluidDrop.computeLevelset())

	flags.updateFromLevelset(phi)
	sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.pause()

	l_ = np.zeros([res_y,res_x], dtype=np.float32)
	for t in trange(args.num_frames):
		v_path_ = os.path.join(v_path, args.path_format % (p1, p2, t))
		with np.load(v_path_) as data:
			v = data['x']
			v = np.dstack((v,np.zeros([res_y, res_x, 1])))

		copyArrayToGridMAC(v, vel)

		markFluidCells(parts=pp, flags=flags)
		extrapolateMACSimple(flags=flags, vel=vel, distance=4)

		gridParticleIndex(parts=pp, flags=flags, indexSys=pindex, index=gpi)
		# unionParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor)
		averagedParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor, 1, 1)
		phi.setBound(1, boundaryWidth=args.bWidth)
		resetOutflow(flags=flags, parts=pp, index=gpi, indexSys=pindex)

		# copy levelset to visualize
		copyGridToArrayLevelset(phi, l_)
		l_file_path = os.path.join(img_dir, '%04d.png' % t)
		l_img = l_[::-1]*255 # yx
		l_img = np.stack((l_img,l_img,l_img), axis=-1).astype(np.uint8)
		l_img = Image.fromarray(l_img)
		l_img.save(l_file_path)

		# extrapolate levelset, needed by particle resampling in adjustNumber / resample
		extrapolateLsSimple(phi=phi, distance=4, inside=True)

		# set source grids for resampling, used in adjustNumber!
		adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=args.min_particles,
					 maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)
		
		pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)
		
		# pt_path = os.path.join(pt_dir, '%04d.uni' % t)
		# pp.save(pt_path)

		s.step()

def main():
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	field_type = ['v'] # 'p'
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
	# l_ = np.zeros([res_y,res_x], dtype=np.float32)
	# p_ = np.zeros([res_y,res_x], dtype=np.float32)
	# s_ = np.zeros([res_y,res_x], dtype=np.float32)

	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# l_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]


	# solver params
	gs = vec3(res_x, res_y, 1)
	gravity = vec3(0,args.gravity,0)

	s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	pressure = s.create(RealGrid)
	# stream = s.create(RealGrid)

	# flip
	velOld = s.create(MACGrid)
	tmpVec3 = s.create(VecGrid)
	
	pp = s.create(BasicParticleSystem) 
	pVel = pp.create(PdataVec3)
	# mesh = s.create(Mesh)
	
	# acceleration data for particle nbs
	pindex = s.create(ParticleIndexSystem) 
	gpi = s.create(IntGrid)

	if (GUI):
		gui = Gui()
		gui.show(True)
		#gui.pause()

	print('start generation')
	for i in trange(len(p_list), desc='scenes'):
		flags.initDomain(boundaryWidth=args.bWidth)
		if args.open_bound:
			setOpenBound(flags, args.bWidth, 'xXyY', FlagOutflow|FlagEmpty)

		vel.clear()
		pressure.clear()
		# stream.clear()
		
		velOld.clear()
		tmpVec3.clear()
	
		pp.clear()
		pVel.clear()

		fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,args.basin_y_pos,1.0)) # basin

		p0, p1 = p_list[i][0], p_list[i][1]
		dropCenter = vec3(p0,args.src_y_pos,0.5)
		dropRadius = p1
		fluidDrop = Sphere(parent=s, center=gs*dropCenter, radius=gs.x*dropRadius)
		phi = fluidBasin.computeLevelset()
		phi.join(fluidDrop.computeLevelset())

		flags.updateFromLevelset(phi)
		sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05)

		fluidVel = Sphere(parent=s, center=gs*dropCenter, radius=gs.x*(dropRadius+0.05))
		fluidSetVel = vec3(0,-1,0)
		
		# set initial velocity
		fluidVel.applyToGrid(grid=vel, value=fluidSetVel)
		mapGridToPartsVec3(source=vel, parts=pp, target=pVel)

		for t in trange(args.num_frames, desc='sim'):
			# FLIP 
			pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)
			
			# make sure we have velocities throught liquid region
			mapPartsToMAC(vel=vel, flags=flags, velOld=velOld, parts=pp, partVel=pVel, weight=tmpVec3) 
			extrapolateMACFromWeight(vel=vel, distance=2, weight=tmpVec3)  # note, tmpVec3 could be free'd now...
			markFluidCells(parts=pp, flags=flags)

			# create approximate surface level set, resample particles
			gridParticleIndex(parts=pp , flags=flags, indexSys=pindex, index=gpi)
			# unionParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor)
			averagedParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor, 1, 1)
			phi.setBound(1, boundaryWidth=args.bWidth)
			resetOutflow(flags=flags, parts=pp, index=gpi, indexSys=pindex) 
			
			# copyGridToArrayLevelset(phi, l_)
			
			# extend levelset somewhat, needed by particle resampling in adjustNumber
			extrapolateLsSimple(phi=phi, distance=4, inside=True); 

			# forces & pressure solve
			addGravity(flags=flags, vel=vel, gravity=gravity)
			setWallBcs(flags=flags, vel=vel)	
			solvePressure(flags=flags, vel=vel, pressure=pressure, phi=phi)
			setWallBcs(flags=flags, vel=vel)
			
			# copyGridToArrayReal(pressure, p_)

			# set source grids for resampling, used in adjustNumber!
			pVel.setSource(vel, isMAC=True)
			adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=args.min_particles, maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)

			# # save before extrapolation
			# copyGridToArrayMAC(vel, v_)

			# make sure we have proper velocities
			extrapolateMACSimple(flags=flags, vel=vel, distance=4)
			flipVelocityUpdate(vel=vel, velOld=velOld, flags=flags, parts=pp, partVel=pVel, flipRatio=0.97)
			
			# save after extrapolation
			copyGridToArrayMAC(vel, v_)
			
			# getStreamfunction(flags=flags, vel=vel, grid=stream)
			# copyGridToArrayReal(target=s_, source=stream)
			
			v_range = [np.minimum(v_range[0], v_.min()),
						np.maximum(v_range[1], v_.max())]
			# l_range = [np.minimum(l_range[0], l_.min()),
			# 		   np.maximum(l_range[1], l_.max())]
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

			s.step()
		gc.collect()
		break

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	# lrange_file = os.path.join(args.log_dir, 'l_range.txt')
	# with open(lrange_file, 'w') as f:
	# 	print('%s: levelset min %.3f max %.3f' % (datetime.now(), l_range[0], l_range[1]))
	# 	f.write('%.3f\n' % l_range[0])
	# 	f.write('%.3f' % l_range[1])

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