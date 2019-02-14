import os
from glob import glob
from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
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
        self.w_num = config.w_size
        self.z_num = config.z_num
        self.dof = int(self.args['num_dof'])
        self.code_path = os.path.join(config.code_path, 'code%d.npz' % self.z_num)
        
        self.features_dim = [None, self.z_num+self.dof] # + x,y
        self.features_w_dim = [None, self.w_num, self.z_num+self.dof]

        self.labels_dim = [None, self.z_num]
        self.labels_w_dim = [None, self.w_num, self.z_num]
        self.batch_size = config.batch_size

        self.features_placeholder = tf.placeholder(tf.float32, self.features_dim)
        self.labels_placeholder = tf.placeholder(tf.float32, self.labels_dim)
        self.features_w_placeholder = tf.placeholder(tf.float32, self.features_w_dim)
        self.labels_w_placeholder = tf.placeholder(tf.float32, self.labels_w_dim)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))\
                    .batch(self.batch_size).repeat().shuffle(buffer_size=50)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))\
                    .batch(self.batch_size)

        train_w_dataset = tf.data.Dataset.from_tensor_slices((self.features_w_placeholder, self.labels_w_placeholder))\
                    .batch(self.batch_size).repeat().shuffle(buffer_size=50)
        
        test_w_dataset = tf.data.Dataset.from_tensor_slices((self.features_w_placeholder, self.labels_w_placeholder))\
                    .batch(self.batch_size)
        
        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()
        self.train_w_iterator = train_w_dataset.make_initializable_iterator()
        self.test_w_iterator = test_w_dataset.make_initializable_iterator()

        # load data
        code = np.load(self.code_path)
        x = code['x']
        y = code['y']
        p = code['p']

        self.num_scenes = code['s']
        self.num_frames = code['f']

        self.code_std = np.std(x)
        y -= x
        self.out_std = np.std(y)
        self.p_std = np.std(p)

        x /= self.code_std
        y /= self.out_std
        p /= self.p_std

        self.x_train = np.concatenate((x,p), axis=-1)
        self.y_train = y

        self.num_train_scenes = int(self.num_scenes * 0.95)
        self.num_test_scenes = self.num_scenes - self.num_train_scenes
        self.num_train = self.num_train_scenes * (self.num_frames-1)
        self.num_test = self.x_train.shape[0] - self.num_train

        self.x_test, self.y_test = self.x_train[self.num_train:], self.y_train[self.num_train:]
        self.x_train, self.y_train = self.x_train[:self.num_train], self.y_train[:self.num_train]
                
        num_batches = self.num_train_scenes * (self.num_frames-self.w_num)
        self.x_train_w = np.zeros(shape=[num_batches]+self.features_w_dim[1:])
        self.y_train_w = np.zeros(shape=[num_batches]+self.labels_w_dim[1:])
        k = 0
        for i in range(self.num_train_scenes):
            for j in range(self.num_frames-self.w_num):
                idx = i*(self.num_frames-1) + j
                # print('%d/%d: %d-%d' % (k, self.x_train_w.shape[0], idx, idx+self.w_num))
                self.x_train_w[k,:,:] = self.x_train[idx:idx+self.w_num,:]
                # self.y_train_w[k,:] = self.y_train[idx+self.w_num-1,:]
                self.y_train_w[k,:,:] = self.y_train[idx:idx+self.w_num,:]
                k += 1

        num_batches = self.num_test_scenes * (self.num_frames-self.w_num)
        self.x_test_w = np.zeros(shape=[num_batches]+self.features_w_dim[1:])
        self.y_test_w = np.zeros(shape=[num_batches]+self.labels_w_dim[1:])
        k = 0
        for i in range(self.num_test_scenes):
            for j in range(self.num_frames-self.w_num):
                idx = i*(self.num_frames-1) + j
                self.x_test_w[k,:,:] = self.x_test[idx:idx+self.w_num,:]
                self.y_test_w[k,:,:] = self.y_test[idx:idx+self.w_num,:]
                k += 1

        self.num_train_w = self.x_train_w.shape[0]
        self.num_test_w = self.x_test_w.shape[0]
        self.num_samples = self.num_train + self.num_test
        
        print('%s: # samples %d (train %d/test %d/batch size %d)' % (
            datetime.now(), self.num_samples, self.num_train, self.num_test, self.batch_size))
        self.train_steps = max(int(self.num_train / self.batch_size + 0.5), 1) # per epoch
        self.test_steps = max(int(self.num_test / self.batch_size + 0.5), 1) # per epoch
        self.train_w_steps = max(int(self.num_train_w / self.batch_size + 0.5), 1) # per epoch
        self.test_w_steps = max(int(self.num_test_w / self.batch_size + 0.5), 1) # per epoch

        self.epochs_per_step = 1 / self.train_w_steps
        self.c_num = 0

    def init_it(self, sess):
        print('%s: initialize train/test dataset iterator' % datetime.now())
                
        self.sess = sess
        self.sess.run(self.train_iterator.initializer,
                 feed_dict={self.features_placeholder: self.x_train,
                            self.labels_placeholder: self.y_train})
        
        self.sess.run(self.train_w_iterator.initializer,
                 feed_dict={self.features_w_placeholder: self.x_train_w,
                            self.labels_w_placeholder: self.y_train_w})

    def init_test_it(self):
        self.sess.run(self.test_iterator.initializer,
                 feed_dict={self.features_placeholder: self.x_test,
                            self.labels_placeholder: self.y_test})

        self.sess.run(self.test_w_iterator.initializer,
                 feed_dict={self.features_w_placeholder: self.x_test_w,
                            self.labels_w_placeholder: self.y_test_w})

    def batch(self, is_window=False):
        if is_window:
            return self.train_w_iterator.get_next()
        else:
            return self.train_iterator.get_next()

    def test_batch(self, is_window=False):
        if is_window:
            return self.test_w_iterator.get_next()
        else:
            return self.test_iterator.get_next()

def main(config):
    prepare_dirs_and_logger(config)
    batch_manager = BatchManager(config)

    # test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)
    
    batch_manager.init_it(sess)
    x, y = batch_manager.batch()
    x_, y_ = sess.run([x, y])
    print(x_.shape, y_.shape)

    x, y = batch_manager.batch(is_window=True)
    x_, y_ = sess.run([x, y])
    print(x_.shape, y_.shape)

    batch_manager.init_test_it()
    x, y = batch_manager.test_batch()
    x_, y_ = sess.run([x, y])
    print(x_.shape, y_.shape)

    x, y = batch_manager.test_batch(is_window=True)
    x_, y_ = sess.run([x, y])
    print(x_.shape, y_.shape)
    
    print('batch manager test done')

if __name__ == "__main__":
    from config import get_config
    from util import prepare_dirs_and_logger, save_config, save_image
    config, unparsed = get_config()

    setattr(config, 'dataset', 'smoke_mov200_f400')
    setattr(config, 'arch', 'nn')
    setattr(config, 'code_path', 'log/smoke_mov200_f400/0208_090808_ae_tag')

    main(config)