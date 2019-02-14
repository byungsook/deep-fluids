from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from model import *
from util import *

class Trainer(object):
    def __init__(self, config, batch_manager):
        self.config = config

        self.batch_manager = batch_manager
        self.x, self.y = batch_manager.batch() # normalized input

        self.is_3d = config.is_3d
        self.dataset = config.dataset
        self.data_type = config.data_type
        self.arch = config.arch

        if 'nn' in self.arch:
            self.xt, self.yt = batch_manager.test_batch()
            self.xtw, self.ytw = batch_manager.test_batch(is_window=True)
            self.xw, self.yw = batch_manager.batch(is_window=True)            
        else:
            if self.is_3d:
                self.x_jaco, self.x_vort = jacobian3(self.x)
            else:
                self.x_jaco, self.x_vort = jacobian(self.x)

        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.c_num = batch_manager.c_num
        self.b_num = config.batch_size
        self.test_b_num = config.test_batch_size

        self.repeat = config.repeat
        self.filters = config.filters
        self.num_conv = config.num_conv
        self.w1 = config.w1
        self.w2 = config.w2
        if 'dg' in self.arch: self.w3 = config.w3

        self.use_c = config.use_curl
        if self.use_c:
            if self.is_3d:
                self.output_shape = get_conv_shape(self.x)[1:-1] + [3]
            else:
                self.output_shape = get_conv_shape(self.x)[1:-1] + [1]
        else:
            self.output_shape = get_conv_shape(self.x)[1:]

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
                
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = config.start_step
        self.step = tf.Variable(self.start_step, name='step', trainable=False)
        # self.max_step = config.max_step
        self.max_step = int(config.max_epoch // batch_manager.epochs_per_step)

        self.lr_update = config.lr_update
        if self.lr_update == 'decay':
            lr_min = config.lr_min
            lr_max = config.lr_max
            self.g_lr = tf.Variable(lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, 
               lr_min+0.5*(lr_max-lr_min)*(tf.cos(tf.cast(self.step, tf.float32)*np.pi/self.max_step)+1), name='g_lr_update')
        elif self.lr_update == 'step':
            self.g_lr = tf.Variable(config.lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr*0.5, config.lr_min), name='g_lr_update')    
        else:
            raise Exception("[!] Invalid lr update method")

        self.lr_update_step = config.lr_update_step
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.save_sec = config.save_sec

        self.is_train = config.is_train
        if 'ae' in self.arch:
            self.z_num = config.z_num
            self.p_num = self.batch_manager.dof
            self.use_sparse = config.use_sparse
            self.sparsity = config.sparsity
            self.w4 = config.w4
            self.w5 = config.w5
            self.code_path = config.code_path
            self.build_model_ae()

        elif 'nn' in self.arch:
            self.z_num = config.z_num
            self.w_num = config.w_size
            self.p_num = self.batch_manager.dof
            self.build_model_nn()

        else:
            self.build_model()

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=self.save_sec,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if 'nn' in self.arch:
            self.batch_manager.init_it(self.sess)
            self.log_step = batch_manager.train_steps
        
        elif self.is_train:
            self.batch_manager.start_thread(self.sess)

        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

    def build_model(self):
        if self.use_c:
            self.G_s, self.G_var = GeneratorBE(self.y, self.filters, self.output_shape, 
                                               num_conv=self.num_conv, repeat=self.repeat)
            self.G_ = curl(self.G_s)
        else:
            self.G_, self.G_var = GeneratorBE(self.y, self.filters, self.output_shape,
                                              num_conv=self.num_conv, repeat=self.repeat)
        self.G = denorm_img(self.G_) # for debug

        self.G_jaco_, self.G_vort_ = jacobian(self.G_)
        self.G_vort = denorm_img(self.G_vort_)
        
        if 'dg' in self.arch:
            # discriminator
            # self.D_x, self.D_var = DiscriminatorPatch(self.x, self.filters)
            # self.D_G, _ = DiscriminatorPatch(self.G_, self.filters, reuse=True)
            D_in = tf.concat([self.x, self.x_vort], axis=-1)
            self.D_x, self.D_var = DiscriminatorPatch(D_in, self.filters)
            G_in = tf.concat([self.G_, self.G_vort_], axis=-1)
            self.D_G, _ = DiscriminatorPatch(G_in, self.filters, reuse=True)
        
        show_all_variables()

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
            g_optimizer = optimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'gd':
            optimizer = tf.train.GradientDescentOptimizer
            g_optimizer = optimizer(self.g_lr)
        else:
            raise Exception("[!] Invalid opimizer")

        # losses
        self.g_loss_l1 = tf.reduce_mean(tf.abs(self.G_ - self.x))
        self.g_loss_j_l1 = tf.reduce_mean(tf.abs(self.G_jaco_ - self.x_jaco))
        self.g_loss = self.g_loss_l1*self.w1 + self.g_loss_j_l1*self.w2
        
        if 'dg' in self.arch:
            self.g_loss_real = tf.reduce_mean(tf.square(self.D_G-1))
            self.d_loss_fake = tf.reduce_mean(tf.square(self.D_G))
            self.d_loss_real = tf.reduce_mean(tf.square(self.D_x-1))

            self.g_loss += self.g_loss_real*self.w3

            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.d_optim = g_optimizer.minimize(self.d_loss, var_list=self.D_var)

        self.g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)
        self.epoch = tf.placeholder(tf.float32)

        # summary
        summary = [
            tf.summary.image("x/G", self.G[:,::-1]),
            tf.summary.image("x/G_vort", self.G_vort[:,::-1]),
            
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_l1", self.g_loss_l1),
            tf.summary.scalar("loss/g_loss_j_l1", self.g_loss_j_l1),

            tf.summary.scalar("misc/epoch", self.epoch),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", self.y),

            tf.summary.scalar("misc/g_lr", self.g_lr),
        ]

        if self.use_c:
            summary += [
                tf.summary.image("G_s", self.G_s[:,::-1]),
            ]

        if 'dg' in self.arch:
            summary += [
                tf.summary.scalar("loss/g_loss_real", tf.sqrt(self.g_loss_real)),
                tf.summary.scalar("loss/d_loss_real", tf.sqrt(self.d_loss_real)),
                tf.summary.scalar("loss/d_loss_fake", tf.sqrt(self.d_loss_fake)),
            ]

        self.summary_op = tf.summary.merge(summary)
        
        summary = [
            tf.summary.image("x/x", denorm_img(self.x)[:,::-1]),
            tf.summary.image("x/vort", denorm_img(self.x_vort)[:,::-1]),
        ]
        self.summary_once = tf.summary.merge(summary) # call just once

    def train(self):
        if 'ae' in self.arch:
            self.train_ae()
        elif 'nn' in self.arch:
            self.train_nn()
        else:
            self.train_()

    def train_(self):
        # test1: varying on each axis
        z_range = [-1, 1]
        z_shape = (self.b_num, self.c_num)
        z_samples = []
        z_varying = np.linspace(z_range[0], z_range[1], num=self.b_num)

        for i in range(self.c_num):
            zi = np.zeros(shape=z_shape)
            zi[:,i] = z_varying
            z_samples.append(zi)

        # test2: compare to gt
        x, pi, zi_ = self.batch_manager.random_list(self.b_num)
        x_w = self.get_vort_image(x/127.5-1)
        x = np.concatenate((x,x_w), axis=0)
        save_image(x, '{}/x_fixed_gt.png'.format(self.model_dir), nrow=self.b_num)

        with open('{}/x_fixed_gt.txt'.format(self.model_dir), 'w') as f:
            f.write(str(pi) + '\n')
            f.write(str(zi_))
        
        zi = np.zeros(shape=z_shape)            
        for i, z_gt in enumerate(zi_):
            zi[i,:] = z_gt
        z_samples.append(zi)

        # call once
        summary_once = self.sess.run(self.summary_once)
        self.summary_writer.add_summary(summary_once, 0)
        self.summary_writer.flush()
        
        # train
        for step in trange(self.start_step, self.max_step):
            if 'dg' in self.arch:
                self.sess.run([self.g_optim, self.d_optim])
            else:
                self.sess.run(self.g_optim)

            if step % self.log_step == 0 or step == self.max_step-1:
                ep = step*self.batch_manager.epochs_per_step
                loss, summary = self.sess.run([self.g_loss,self.summary_op],
                                              feed_dict={self.epoch: ep})
                assert not np.isnan(loss), 'Model diverged with loss = NaN'
                print("\n[{}/{}/ep{:.2f}] Loss: {:.6f}".format(step, self.max_step, ep, loss))

                self.summary_writer.add_summary(summary, global_step=step)
                self.summary_writer.flush()

            if step % self.test_step == 0 or step == self.max_step-1:
                self.generate(z_samples, self.model_dir, idx=step)

            if self.lr_update == 'step':
                if step % self.lr_update_step == self.lr_update_step - 1:
                    self.sess.run(self.g_lr_update)
            else:
                self.sess.run(self.g_lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()

    def build_test_model(self):
        # build a model for testing
        self.z = tf.placeholder(dtype=tf.float32, shape=[self.test_b_num, self.c_num])
        if self.use_c:
            self.G_s, _ = GeneratorBE(self.z, self.filters, self.output_shape,
                                      num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            self.G_ = curl(self.G_s)
        else:
            self.G_, _ = GeneratorBE(self.z, self.filters, self.output_shape,
                                     num_conv=self.num_conv, repeat=self.repeat, reuse=True)

    def test(self):
        if 'ae' in self.arch:
            self.test_ae()
        elif 'nn' in self.arch:
            self.test_nn()
        else:
            self.test_()

    def test_(self):
        self.build_test_model()
        
        p1, p2 = 10, 2

        # eval
        y1 = int(self.batch_manager.y_num[0])
        y2 = int(self.batch_manager.y_num[1])
        y3 = int(self.batch_manager.y_num[2])

        assert(y3 % self.test_b_num == 0)
        niter = int(y3 / self.test_b_num)

        c1 = p1/float(y1-1)*2-1
        c2 = p2/float(y2-1)*2-1

        z_range = [-1, 1]
        z_varying = np.linspace(z_range[0], z_range[1], num=y3)
        z_shape = (y3, self.c_num)

        z_c = np.zeros(shape=z_shape)
        z_c[:,0] = c1
        z_c[:,1] = c2
        z_c[:,-1] = z_varying

        G = []
        for b in range(niter):
            G_ = self.sess.run(self.G_, {self.z: z_c[self.test_b_num*b:self.test_b_num*(b+1),:]})
            G_, _ = self.batch_manager.denorm(x=G_)
            G.append(G_)
        G = np.concatenate(G, axis=0)

        # save
        title = '%d_%d' % (p1,p2)
        out_dir = os.path.join(self.model_dir, title)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i, G_ in enumerate(G):
            dump_path = os.path.join(out_dir, '%d.npz' % i)
            np.savez_compressed(dump_path, x=G_)


    def build_model_ae(self):
        if self.use_c:
            self.s, self.z, self.var = AE(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                          num_conv=self.num_conv, repeat=self.repeat)
            self.x_ = curl(self.s)
        else:
            self.x_, self.z, self.var = AE(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                              num_conv=self.num_conv, repeat=self.repeat)
        self.x_img = denorm_img(self.x_) # for debug

        self.x_jaco_, self.x_vort_ = jacobian(self.x_)
        self.x_vort_ = denorm_img(self.x_vort_)

        show_all_variables()

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
            g_optimizer = optimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == 'gd':
            optimizer = tf.train.GradientDescentOptimizer
            g_optimizer = optimizer(self.g_lr)
        else:
            raise Exception("[!] Invalid opimizer")

        # losses
        self.loss_l1 = tf.reduce_mean(tf.abs(self.x_ - self.x))
        self.loss_j_l1 = tf.reduce_mean(tf.abs(self.x_jaco_ - self.x_jaco))
        
        y = self.y[:,:,-1]
        self.loss_p = tf.reduce_mean(tf.squared_difference(y, self.z[:,-self.p_num:]))
        self.loss = self.loss_l1*self.w1 + self.loss_j_l1*self.w2 + self.loss_p*self.w4
        
        if self.use_sparse:
            ds = tf.distributions
            rho = ds.Bernoulli(probs=self.sparsity)
            rho_ = ds.Bernoulli(probs=tf.reduce_mean(self.z[:,:-self.p_num], axis=0))
            self.loss_kl = tf.reduce_sum(ds.kl_divergence(rho, rho_))
            self.loss += self.loss_kl*self.w5

        self.optim = g_optimizer.minimize(self.loss, global_step=self.step, var_list=self.var)
        self.epoch = tf.placeholder(tf.float32)

        # summary
        summary = [
            tf.summary.image("x", self.x_img[:,::-1]),
            tf.summary.image("x_vort", self.x_vort_[:,::-1]),
            
            tf.summary.scalar("loss/total_loss", self.loss),
            tf.summary.scalar("loss/loss_l1", self.loss_l1),
            tf.summary.scalar("loss/loss_j_l1", self.loss_j_l1),
            tf.summary.scalar("loss/loss_p", self.loss_p),

            tf.summary.scalar("misc/epoch", self.epoch),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", y),
            tf.summary.histogram("z", self.z),

            tf.summary.scalar("misc/g_lr", self.g_lr),
        ]

        if self.use_sparse:
            summary += [
                tf.summary.scalar("loss/loss_kl", self.loss_kl),
            ]

        self.summary_op = tf.summary.merge(summary)

    def train_ae(self):
        x, pi, zi_ = self.batch_manager.random_list(self.b_num)
        x_w = self.get_vort_image(x/127.5-1)
        x = np.concatenate((x,x_w), axis=0)
        save_image(x, '{}/x_fixed_gt.png'.format(self.model_dir), nrow=self.b_num)

        with open('{}/x_fixed_gt.txt'.format(self.model_dir), 'w') as f:
            f.write(str(pi) + '\n')
            f.write(str(zi_))        
        
        # train
        for step in trange(self.start_step, self.max_step):
            self.sess.run(self.optim)

            if step % self.log_step == 0 or step == self.max_step-1:
                ep = step*self.batch_manager.epochs_per_step
                loss, summary = self.sess.run([self.loss,self.summary_op],
                                              feed_dict={self.epoch: ep})

                assert not np.isnan(loss), 'Model diverged with loss = NaN'
                print("\n[{}/{}/ep{:.2f}] Loss: {:.6f}".format(step, self.max_step, ep, loss))

                self.summary_writer.add_summary(summary, global_step=step)
                self.summary_writer.flush()

            if step % self.test_step == 0 or step == self.max_step-1:
                self.autoencode(x, self.model_dir, idx=step)

            if self.lr_update == 'step':
                if step % self.lr_update_step == self.lr_update_step - 1:
                    self.sess.run(self.g_lr_update)
            else:
                self.sess.run(self.g_lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()

    def build_test_model_ae(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.test_b_num, self.res_y, self.res_x, 2])
        if self.use_c:
            self.s, self.z, self.var = AE(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                          num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            self.x_ = curl(self.s)
        else:
            self.x_, self.z, self.var = AE(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                              num_conv=self.num_conv, repeat=self.repeat, reuse=True)
        self.x_img = denorm_img(self.x_)

    def test_ae(self):
        self.build_test_model_ae()

        if not self.code_path:
            # dump latent codes
            n_path = os.path.join(self.batch_manager.root, 'n.npz')
            with np.load(n_path) as data:
                nx = data['nx']
                if self.is_3d: nz = data['nz']

            num_sims = nx.shape[0] # 5
            num_frames = nx.shape[1]
            # print(num_sims, num_frames)

            if self.is_3d:
                dx_list = (nx[:,1:] - nx[:,:-1]).reshape([-1, 1])
                dz_list = (nz[:,1:] - nz[:,:-1]).reshape([-1, 1])
                p_list = np.concatenate((dx_list,dz_list), axis=-1)
            else:
                dx_list = (nx[:,1:] - nx[:,:-1]).reshape([-1, 1])
                p_list = dx_list

            from tqdm import tqdm            
            c_list = []
            num_iter = num_sims*num_frames/self.test_b_num
            for x, _ in tqdm(self.batch_manager.batch_(self.test_b_num),
                            total=num_iter):
                c = self.sess.run(self.z, {self.x: x})
                c_list.append(c)

            c_list = np.concatenate(c_list)
            x_list = []
            y_list = []
            
            for i in range(num_sims):
                s1 = i*num_frames
                s2 = (i+1)*num_frames
                x_list.append(c_list[s1:s2-1,:])
                y_list.append(c_list[s1+1:s2,:])
            
            x_list = np.concatenate(x_list)
            y_list = np.concatenate(y_list)        
            print(x_list.shape, y_list.shape, p_list.shape)

            code_path = os.path.join(self.load_path, 'code%d.npz' % self.z_num)
            np.savez_compressed(code_path,
                                x=x_list, 
                                y=y_list,
                                p=p_list,
                                s=num_sims,
                                f=num_frames)
        else:
            # reconstruct velocity from latent codes
            code_path = os.path.join(self.code_path, 'code_out.npz')
            with np.load(code_path) as data:
                z_ = data['z_out']
                z_gt_ = data['z_gt']
            
            num_sims = z_.shape[0]
            num_frames = z_[0].shape[0]
            num_iters = int(num_frames / self.test_b_num)

            if self.is_3d:
                x_img = self.x_img['xym']
            else:
                x_img = self.x_img

            for s in range(num_sims):
                z = z_[s]
                z_gt = z_gt_[s]

                generated = []
                v = None
                v_gt = None

                for i in trange(num_iters):
                    v_, x_img_ = self.sess.run([self.x_, x_img], {self.z: z[i*self.test_b_num:(i+1)*self.test_b_num,:]})
                    v_gt_, x_img_gt = self.sess.run([self.x_, x_img], {self.z: z_gt[i*self.test_b_num:(i+1)*self.test_b_num,:]})
                    v_, _ = self.batch_manager.denorm(v_)
                    v_gt_, _ = self.batch_manager.denorm(v_gt_)

                    if v is None:
                        v = v_
                    else:
                        v = np.concatenate((v, v_), axis=0)

                    if v_gt is None:
                        v_gt = v_gt_
                    else:
                        v_gt = np.concatenate((v_gt, v_gt_), axis=0)

                    for j in range(self.test_b_num):
                        generated.append(x_img_[j])
                        generated.append(x_img_gt[j])

                generated = np.asarray(generated)
                # print(generated.shape, v.shape)

                img_dir = os.path.join(self.load_path, 'img%d' % s)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                for i in range(num_frames):
                    img_path = os.path.join(img_dir, str(i)+'.png')
                    save_image(generated[i*2:i*2+2], img_path, nrow=2)

                # save v
                # v_path = os.path.join(self.load_path, 'v%d.npz' % s)
                # np.savez_compressed(v_path, v=v, v_gt=v_gt)


    def build_model_nn(self):
        self.y_, self.var = NN(self.x, self.filters, self.z_num)
        self.yt_, _ = NN(self.xt, self.filters, self.z_num, train=False, reuse=True)

        x_ = self.xw[:,0,:]
        xt_ = self.xtw[:,0,:]
        yw_ = None
        ytw_ = None
        for i in range(self.w_num):
            y_, _ = NN(x_, self.filters, self.z_num, reuse=True)
            yt_, _ = NN(xt_, self.filters, self.z_num, train=False, reuse=True)
            yw = tf.expand_dims(y_, 1)
            yt = tf.expand_dims(yt_, 1)
            if yw_ is None:
                yw_ = yw
            else:
                yw_ = tf.concat((yw_, yw), axis=1)
            if ytw_ is None:
                ytw_ = yt
            else:
                ytw_ = tf.concat((ytw_, yt), axis=1)

            if i < self.w_num-1:
                # re-normalized to the scale of x
                y_ *= (self.batch_manager.out_std / self.batch_manager.code_std)
                yt_ *= (self.batch_manager.out_std / self.batch_manager.code_std)
                
                x_ = tf.concat([tf.add(x_[:,:-self.p_num], y_), self.xw[:,i+1,-self.p_num:]], axis=-1)
                xt_ = tf.concat([tf.add(xt_[:,:-self.p_num], yt_), self.xtw[:,i+1,-self.p_num:]], axis=-1)

        self.yw_ = yw_
        self.ytw_ = ytw_

        show_all_variables()

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(self.optimizer))

        optimizer = optimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        
        # losses
        self.loss_train = tf.losses.mean_squared_error(self.y, self.y_)
        self.loss_train_w = tf.losses.mean_squared_error(self.yw, self.yw_)
        # self.loss = self.loss_train*self.w1 + self.loss_train_w*self.w2
        self.loss = self.loss_train_w
        # self.loss = self.loss_train

        self.l_test = tf.losses.mean_squared_error(self.yt, self.yt_)
        self.loss_test = tf.placeholder(tf.float32)
        self.l_test_w = tf.losses.mean_squared_error(self.ytw, self.ytw_)
        self.loss_test_w = tf.placeholder(tf.float32)

        self.optim = optimizer.minimize(self.loss, global_step=self.step, var_list=self.var)
        self.epoch = tf.placeholder(tf.float32)

        # summary
        summary = [
            tf.summary.scalar("loss/total_loss", self.loss),
            tf.summary.scalar("loss/loss_train", self.loss_train),
            tf.summary.scalar("loss/loss_test", self.loss_test),
            tf.summary.scalar("loss/loss_train_w", self.loss_train_w),
            tf.summary.scalar("loss/loss_test_w", self.loss_test_w),
            
            tf.summary.scalar("misc/lr", self.g_lr),
            tf.summary.scalar("misc/epoch", self.epoch),
        ]

        self.summary_op = tf.summary.merge(summary)

    def train_nn(self):
        # train
        ep = 0
        for step in trange(self.start_step, self.max_step):
            self.sess.run(self.optim)

            if step % self.log_step == 0 or step == self.max_step-1:
                ep += 1

                self.batch_manager.init_test_it()
                test_loss = 0.0
                for t in range(self.batch_manager.test_steps):
                    tl = self.sess.run(self.l_test)
                    test_loss += tl
                test_loss /= self.batch_manager.test_steps

                test_loss_w = 0.0
                for t in range(self.batch_manager.test_w_steps):
                    tl = self.sess.run(self.l_test_w)
                    test_loss_w += tl
                test_loss_w /= self.batch_manager.test_w_steps

                loss, summary = self.sess.run([self.loss,self.summary_op],
                    feed_dict={self.epoch: ep, self.loss_test: test_loss,
                               self.loss_test_w: test_loss_w})
                assert not np.isnan(loss), 'Model diverged with loss = NaN'
                # print("\n[{}/{}/ep{:.2f}] Loss: {:.6f}/{:.6f}".format(step, self.max_step, ep, loss, test_loss))

                self.summary_writer.add_summary(summary, global_step=step)
                self.summary_writer.flush()

            if self.lr_update == 'step':
                if step % self.lr_update_step == self.lr_update_step - 1:
                    self.sess.run(self.g_lr_update)
            else:
                self.sess.run(self.g_lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)

    def test_nn(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[1, self.z_num+self.p_num])
        self.y, _ = NN(self.x, self.filters, self.z_num, train=False, reuse=True)

        z_out_list = []
        z_gt_list = []
        num_sims = self.batch_manager.num_test_scenes
        num_frames = self.batch_manager.num_frames

        for i in range(num_sims):
            z0 = self.batch_manager.x_test[i*(num_frames-1)]
            z_in = z0.reshape(1,-1)
            z_out = [z0[:-self.p_num].reshape(1,-1)*self.batch_manager.code_std]
            z_gt = [z0[:-self.p_num].reshape(1,-1)*self.batch_manager.code_std]
            for t in range(num_frames-1):
                y_gt = self.batch_manager.y_test[i*(num_frames-1)+t]*self.batch_manager.out_std +\
                    self.batch_manager.x_test[i*(num_frames-1)+t,:-self.p_num]*self.batch_manager.code_std
                z_gt.append(y_gt.reshape(1, -1))

                y_ = self.sess.run(self.y, {self.x: z_in})*self.batch_manager.out_std +\
                    z_in[:,:-self.p_num]*self.batch_manager.code_std
                y_[0,-self.p_num:] = y_gt[-self.p_num:] # px, py
                z_out.append(y_)

                if t < num_frames-2:
                    zt = self.batch_manager.x_test[i*(num_frames-1)+t+1]
                    # z_in = self.batch_manager.x_test[i*149+t+1].reshape(1, -1) # gt..
                    z_in = np.append(y_.flatten()/self.batch_manager.code_std,zt[-self.p_num:]).reshape(1,-1)

            z_out_list.append(np.concatenate(z_out))
            z_gt_list.append(np.concatenate(z_gt))

        z_out = np.stack(z_out_list)
        z_gt = np.stack(z_gt_list)

        # z_diff = np.mean(abs(z_out[0] - z_gt[0]), axis=-1)
        # import matplotlib.pyplot as plt
        # plt.plot(range(num_frames), z_diff)
        # plt.show()
        # fig_path = os.path.join(self.load_path, 'code_diff0.png')
        # plt.savefig(fig_path)

        # diff_path = os.path.join(self.load_path, '%s.npz' % self.load_path.split('/')[-1])
        # np.savez_compressed(diff_path, z=z_diff)
        # # exit()
        
        code_path = os.path.join(self.load_path, 'code_out.npz')
        np.savez_compressed(code_path,
							z_out=z_out, 
                            z_gt=z_gt)

    
    def generate(self, inputs, root_path=None, idx=None):
        generated = []
        for i, z_sample in enumerate(inputs):
            generated.append(self.sess.run(self.G, {self.y: z_sample}))
            
        c_concat = np.concatenate(tuple(generated[:-1]), axis=0)
        c_path = os.path.join(root_path, '{}_c.png'.format(idx))
        save_image(c_concat, c_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(c_path))

        c_vort = self.get_vort_image(c_concat/127.5-1)
        c_path = os.path.join(root_path, '{}_cv.png'.format(idx))
        save_image(c_vort, c_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(c_path))

        x = generated[-1]
        x_path = os.path.join(root_path, 'x_fixed_{}.png'.format(idx))
        x_w = self.get_vort_image(x/127.5-1)
        x = np.concatenate((x,x_w), axis=0)

        save_image(x, x_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(x_path))

    def get_vort_image(self, x):
        x = vort_np(x[:,:,:,:2])
        if not 'ae' in self.arch: x /= np.abs(x).max() # [-1,1]
        x_img = (x+1)*127.5
        x_img = np.uint8(plt.cm.RdBu(x_img[...,0]/255)*255)[...,:3]
        return x_img

    def autoencode(self, inputs, root_path=None, idx=None):
        # only for 2d
        inputs = inputs[:self.b_num,...,:-1] # take vort off
        x_gt = inputs/127.5 - 1 # to [-1,1]
        x = self.sess.run(self.x_img, {self.x: x_gt})
        x_w = self.get_vort_image(x/127.5-1)
        x = np.concatenate((x,x_w), axis=0)

        x_path = os.path.join(root_path, '{}.png'.format(idx))
        save_image(x, x_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(x_path))