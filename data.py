import os
from glob import glob

import threading
import multiprocessing
import signal
import sys
from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ops import *

class BatchManager(object):
    def __init__(self, config):
        self.rng = np.random.RandomState(config.random_seed)
        self.root = config.data_path        

        # read data generation arguments
        self.args = {}
        with open(os.path.join(self.root, 'args.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self.args[arg] = arg_value

        self.is_3d = config.is_3d
        if 'ae' in config.arch:
            def sortf(x):
                nf = int(self.args['num_frames'])
                n = os.path.basename(x)[:-4].split('_')
                return int(n[0])*nf + int(n[1])

            self.paths = sorted(glob("{}/{}/*".format(self.root, config.data_type[0])),
                                key=sortf)
            # num_path = len(self.paths)          
            # num_train = int(num_path*0.95)
            # self.test_paths = self.paths[num_train:]
            # self.paths = self.paths[:num_train]
        else:
            self.paths = sorted(glob("{}/{}/*".format(self.root, config.data_type[0])))
        
        self.num_samples = len(self.paths)
        assert(self.num_samples > 0)
        self.batch_size = config.batch_size
        self.epochs_per_step = self.batch_size / float(self.num_samples) # per epoch

        self.data_type = config.data_type
        if self.data_type == 'velocity':
            if self.is_3d: depth = 3
            else: depth = 2
        else:
            depth = 1
        
        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.depth = depth
        self.c_num = int(self.args['num_param'])

        if self.is_3d:
            feature_dim = [self.res_z, self.res_y, self.res_x, self.depth]
        else:
            feature_dim = [self.res_y, self.res_x, self.depth]
        
        if 'ae' in config.arch:
            self.dof = int(self.args['num_dof'])
            label_dim = [self.dof, int(self.args['num_frames'])]
        else:
            label_dim = [self.c_num]

        if self.is_3d:
            min_after_dequeue = 500
        else:
            min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * self.batch_size
        self.q = tf.FIFOQueue(capacity, [tf.float32, tf.float32], [feature_dim, label_dim])
        self.x = tf.placeholder(dtype=tf.float32, shape=feature_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=label_dim)
        self.enqueue = self.q.enqueue([self.x, self.y])
        self.num_threads = np.amin([config.num_worker, multiprocessing.cpu_count(), self.batch_size])

        r = np.loadtxt(os.path.join(self.root, self.data_type[0]+'_range.txt'))
        self.x_range = max(abs(r[0]), abs(r[1]))
        self.y_range = []
        self.y_num = []

        if 'ae' in config.arch:
            for i in range(self.c_num):
                p_name = self.args['p%d' % i]
                p_min = float(self.args['min_{}'.format(p_name)])
                p_max = float(self.args['max_{}'.format(p_name)])
                p_num = int(self.args['num_{}'.format(p_name)])
                self.y_num.append(p_num)
            for i in range(label_dim[0]):
                self.y_range.append([-1, 1])
        else:
            for i in range(self.c_num):
                p_name = self.args['p%d' % i]
                p_min = float(self.args['min_{}'.format(p_name)])
                p_max = float(self.args['max_{}'.format(p_name)])
                p_num = int(self.args['num_{}'.format(p_name)])
                self.y_range.append([p_min, p_max])
                self.y_num.append(p_num)

    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass

    def start_thread(self, sess):
        print('%s: start to enque with %d threads' % (datetime.now(), self.num_threads))

        # Main thread: create a coordinator.
        self.sess = sess
        self.coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, paths, rng,
                           x, y, data_type, x_range, y_range):
            with coord.stop_on_exception():                
                while not coord.should_stop():
                    id = rng.randint(len(paths))
                    x_, y_ = preprocess(paths[id], data_type, x_range, y_range)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self.threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(self.sess, 
                                                self.enqueue,
                                                self.coord,
                                                self.paths,
                                                self.rng,
                                                self.x,
                                                self.y,
                                                self.data_type,
                                                self.x_range,
                                                self.y_range)
                                          ) for i in range(self.num_threads)]

        # define signal handler
        def signal_handler(signum, frame):
            #print "stop training, save checkpoint..."
            #saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            print('%s: canceled by SIGINT' % datetime.now())
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self.threads:
            t.start()

    def stop_thread(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.coord.request_stop()
        self.sess.run(self.q.close(cancel_pending_enqueues=True))
        self.coord.join(self.threads)

    def batch(self):
        return self.q.dequeue_many(self.batch_size)

    def batch_(self, b_num):
        assert(len(self.paths) % b_num == 0)
        x_batch = []
        y_batch = []
        for i, filepath in enumerate(self.paths):
            x, _ = preprocess(filepath, self.data_type, self.x_range, self.y_range)
            x_batch.append(x)

            if (i+1) % b_num == 0:
                yield np.array(x_batch), y_batch
                x_batch.clear()
                y_batch.clear()

    def denorm(self, x=None, y=None):
        # input range [-1, 1] -> original range
        if x is not None:
            x *= self.x_range

        if y is not None:
            r = self.y_range
            for i, ri in enumerate(self.y_range):
                y[:,i] = (y[:,i]+1) * 0.5 * (ri[1]-ri[0]) + ri[0]
        return x, y

    def list_from_p(self, p_list):
        path_format = os.path.join(self.root, self.data_type[0], self.args['path_format'])
        filelist = []
        for p in p_list:
            filelist.append(path_format % tuple(p))
        return filelist

    def random_list2d(self, num):
        xs = []
        pis = []
        zis = []
        for _ in range(num):
            pi = []
            for y_max in self.y_num:
                pi.append(self.rng.randint(y_max))

            filepath = self.list_from_p([pi])[0]
            x, y = preprocess(filepath, self.data_type, self.x_range, self.y_range)
            if self.data_type[0] == 'v':
                b_ch = np.zeros((self.res_y, self.res_x, 1))
                x = np.concatenate((x, b_ch), axis=-1)
            elif self.data_type[0] == 'l':
                offset = 0.5
                eps = 1e-3
                x[x<(offset+eps)] = -1
                x[x>-1] = 1
            x = np.clip((x+1)*127.5, 0, 255)
            zi = [(p/float(self.y_num[i]-1))*2-1 for i, p in enumerate(pi)] # [-1,1]

            xs.append(x)
            pis.append(pi)
            zis.append(zi)
        return np.array(xs), pis, zis

    def random_list3d(self, num):
        sample = {
            'x': [],
            'y': [],
            'xy': [],
            'zy': [],
            'xym': [],
            'zym': [],
            'xy_c': [],
            'zy_c': [],
            'xym_c': [],
            'zym_c': [],
            'xy_w': [],
            'zy_w': [],
            'xym_w': [],
            'zym_w': [],
            'p': [],
            'z': [],
        }
        
        for _ in range(num):
            p = []
            for y_max in self.y_num:
                p.append(self.rng.randint(y_max))
            sample['p'].append(p)
            z = [(pi/float(self.y_num[i]-1))*2-1 for i, pi in enumerate(p)] # [-1,1]
            sample['z'].append(z)

            file_path = self.list_from_p([p])[0]
            x, y = preprocess(file_path, self.data_type, self.x_range, self.y_range)
            sample['x'].append(x)
            sample['y'].append(y)

            xy = plane_view_np(x, xy_plane=True, project=True)
            zy = plane_view_np(x, xy_plane=False, project=True)
            xym = plane_view_np(x, xy_plane=True, project=False)
            zym = plane_view_np(x, xy_plane=False, project=False)

            sample['xy'].append(xy)
            sample['zy'].append(zy)
            sample['xym'].append(xym)
            sample['zym'].append(zym)

            # vorticity
            x_c = np.expand_dims(x, axis=0)
            _, x_c = jacobian_np3(x_c)

            x_c = np.squeeze(x_c, axis=0)
            xy_c = plane_view_np(x_c, xy_plane=True, project=True)
            zy_c = plane_view_np(x_c, xy_plane=False, project=True)
            xym_c = plane_view_np(x_c, xy_plane=True, project=False)
            zym_c = plane_view_np(x_c, xy_plane=False, project=False)

            sample['xy_c'].append(xy_c)
            sample['zy_c'].append(zy_c)
            sample['xym_c'].append(xym_c)
            sample['zym_c'].append(zym_c)
            
        sample['x'] = np.array(sample['x'])
        sample['y'] = np.array(sample['y'])

        sample['xy'] = np.array(sample['xy'])
        sample['zy'] = np.array(sample['zy'])
        sample['xym'] = np.array(sample['xym'])
        sample['zym'] = np.array(sample['zym'])

        sample['xy_c'] = np.array(sample['xy_c'])
        sample['zy_c'] = np.array(sample['zy_c'])
        sample['xym_c'] = np.array(sample['xym_c'])
        sample['zym_c'] = np.array(sample['zym_c'])

        return sample

    def random_list(self, num):
        if self.is_3d:
            return self.random_list3d(num)
        else:
            return self.random_list2d(num)
    

def preprocess(file_path, data_type, x_range, y_range):    
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']

    # # ############## for old data
    # if x.ndim == 4:
    #     x = x.transpose([2,0,1,3]) # yxzd -> zyxd
    # else:
    #     y = y[None,]
    #     x = x[:,::-1] # horizontal flip
    # else:
    #     x = x[::-1] # horizontal flip

    # normalize
    if data_type[0] == 'd':
        x = x*2 - 1
    else:
        x /= x_range
        
    for i, ri in enumerate(y_range):
        y[i] = (y[i]-ri[0]) / (ri[1]-ri[0]) * 2 - 1
    return x, y

def test3d(config):
    prepare_dirs_and_logger(config)
    tf.set_random_seed(config.random_seed)

    batch_manager = BatchManager(config)

    # batch test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)

    batch_manager.start_thread(sess)
    x, y = batch_manager.batch()
    x_ = x.eval(session=sess)    
    batch_manager.stop_thread()
    
    x_ = (x_+1)*127.5 # [0, 255]
    x_ = np.mean(x_, axis=1) # yx
    save_image(x_, '{}/x_fixed.png'.format(config.model_dir))

    # random pick from parameter space
    sample = batch_manager.random_list(config.batch_size)    
    save_image(sample['xy'], '{}/xy.png'.format(config.model_dir))
    save_image(sample['zy'], '{}/zy.png'.format(config.model_dir))
    save_image(sample['xym'], '{}/xym.png'.format(config.model_dir))
    save_image(sample['zym'], '{}/zym.png'.format(config.model_dir))
    with open('{}/sample.txt'.format(config.model_dir), 'w') as f:
        f.write(str(sample['p']))
        f.write(str(sample['z']))

def test2d(config):
    prepare_dirs_and_logger(config)
    tf.set_random_seed(config.random_seed)

    batch_manager = BatchManager(config)

    # thread test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)
    batch_manager.start_thread(sess)

    x, y = batch_manager.batch() # [-1, 1]
    x_ = x.eval(session=sess)
    # y_ = y.eval(session=sess)
    batch_manager.stop_thread()

    x_w = vort_np(x_)
    x_w /= np.abs(x_w).max()
    x_w = (x_w+1)*0.5
    x_w = np.uint8(plt.cm.RdBu(x_w[...,0])*255)[...,:3]
    x_ = (x_+1)*127.5 # [0, 255]
    b_ch = np.ones([config.batch_size,config.res_y,config.res_x,1])*127.5
    x_ = np.concatenate((x_, b_ch), axis=-1)
    x_ = np.concatenate((x_, x_w), axis=0)
    save_image(x_, '{}/x_fixed.png'.format(config.model_dir))

    # random pick from parameter space
    x, pi, zi = batch_manager.random_list(config.batch_size)
    x_w = vort_np(x/127.5-1)
    x_w /= np.abs(x_w).max()
    x_w = (x_w+1)*0.5
    x_w = np.uint8(plt.cm.RdBu(x_w[...,0])*255)[...,:3]
    x = np.concatenate((x, x_w), axis=0)
    save_image(x, '{}/x.png'.format(config.model_dir))
    with open('{}/x_p.txt'.format(config.model_dir), 'w') as f:
        f.write(str(pi))
        f.write(str(zi))

if __name__ == "__main__":
    from config import get_config
    from util import prepare_dirs_and_logger, save_config, save_image
    config, unparsed = get_config()

    # ##############
    # test: 2d
    setattr(config, 'dataset', 'smoke_pos21_size5_f200')
    setattr(config, 'res_x', 96)
    setattr(config, 'res_y', 128)

    # setattr(config, 'dataset', 'liquid_pos10_size4_f200')
    # setattr(config, 'res_x', 128)
    # setattr(config, 'res_y', 64)

    # setattr(config, 'dataset', 'smoke_rot_f500')
    # setattr(config, 'res_x', 96)
    # setattr(config, 'res_y', 128)
    # setattr(config, 'arch', 'ae')

    # setattr(config, 'dataset', 'smoke_mov200_f400')
    # setattr(config, 'res_x', 96)
    # setattr(config, 'res_y', 128)
    # setattr(config, 'arch', 'ae')

    test2d(config)

    # ##############
    # # test: 3d
    # # setattr(config, 'is_3d', True)
    # # setattr(config, 'batch_size', 4)

    # # setattr(config, 'dataset', 'smoke3_vel5_buo3_f250')
    # # setattr(config, 'res_x', 112)
    # # setattr(config, 'res_y', 64)
    # # setattr(config, 'res_z', 32)

    # # setattr(config, 'dataset', 'smoke3_obs11_buo4_f150')
    # # setattr(config, 'res_x', 64)
    # # setattr(config, 'res_y', 96)
    # # setattr(config, 'res_z', 64)

    # # setattr(config, 'dataset', 'liquid3_d5_r10_f150')
    # # setattr(config, 'res_x', 96)
    # # setattr(config, 'res_y', 48)
    # # setattr(config, 'res_z', 96)

    # # setattr(config, 'dataset', 'liquid3_vis4_f150')
    # # setattr(config, 'res_x', 96)
    # # setattr(config, 'res_y', 72)
    # # setattr(config, 'res_z', 48)

    # # setattr(config, 'dataset', 'smoke3_rot_f500')
    # # setattr(config, 'res_x', 48)
    # # setattr(config, 'res_y', 72)
    # # setattr(config, 'res_z', 48)
    # # setattr(config, 'arch', 'ae')

    # # setattr(config, 'dataset', 'smoke3_mov200_f400')
    # # setattr(config, 'res_x', 48)
    # # setattr(config, 'res_y', 72)
    # # setattr(config, 'res_z', 48)
    # # setattr(config, 'arch', 'ae')

    # test3d(config)
    # ##############