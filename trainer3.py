from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from datetime import datetime
import time

from model import *
from util import *
from trainer import Trainer

class Trainer3(Trainer):
    def build_model(self):
        if self.use_c:
            self.G_s, self.G_var = GeneratorBE3(self.y, self.filters, self.output_shape, 
                                               num_conv=self.num_conv, repeat=self.repeat)
            _, self.G_ = jacobian3(self.G_s)
        else:
            self.G_, self.G_var = GeneratorBE3(self.y, self.filters, self.output_shape,
                                              num_conv=self.num_conv, repeat=self.repeat)
        self.G = denorm_img3(self.G_) # for debug

        self.G_jaco_, self.G_vort_ = jacobian3(self.G_)
        self.G_vort = denorm_img3(self.G_vort_)
        
        if 'dg' in self.arch:
            # discriminator
            # self.D_x, self.D_var = DiscriminatorPatch3(self.x, self.filters)
            # self.D_G, _ = DiscriminatorPatch3(self.G_, self.filters, reuse=True)
            D_in = tf.concat([self.x, self.x_vort], axis=-1)
            self.D_x, self.D_var = DiscriminatorPatch3(D_in, self.filters)
            G_in = tf.concat([self.G_, self.G_vort_], axis=-1)
            self.D_G, _ = DiscriminatorPatch3(G_in, self.filters, reuse=True)


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
            # tf.summary.image("xy/G", self.G['xy']),
            # tf.summary.image("zy/G", self.G['zy']),
            tf.summary.image("xym/G", self.G['xym'][:,::-1]),
            tf.summary.image("zym/G", self.G['zym'][:,::-1]),
            
            # tf.summary.image("xy/G_vort", self.G_vort['xy']),
            # tf.summary.image("zy/G_vort", self.G_vort['zy']),
            tf.summary.image("xym/G_vort", self.G_vort['xym'][:,::-1]),
            tf.summary.image("zym/G_vort", self.G_vort['zym'][:,::-1]),
            
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_l1", self.g_loss_l1),
            tf.summary.scalar("loss/g_loss_j_l1", self.g_loss_j_l1),

            tf.summary.scalar("misc/epoch", self.epoch),
            tf.summary.scalar('misc/q', self.batch_manager.q.size()),

            tf.summary.histogram("y", self.y),

            tf.summary.scalar("misc/g_lr", self.g_lr),
        ]

        if 'dg' in self.arch:
            summary += [
                tf.summary.scalar("loss/g_loss_real", tf.sqrt(self.g_loss_real)),
                tf.summary.scalar("loss/d_loss_real", tf.sqrt(self.d_loss_real)),
                tf.summary.scalar("loss/d_loss_fake", tf.sqrt(self.d_loss_fake)),
            ]

        self.summary_op = tf.summary.merge(summary)

        # summary once
        x = denorm_img3(self.x)
        x_vort = denorm_img3(self.x_vort)
        
        summary = [
            tf.summary.image("xym/x", x['xym'][:,::-1]),
            tf.summary.image("zym/x", x['zym'][:,::-1]),
            tf.summary.image("xym/vort", x_vort['xym'][:,::-1]),
            tf.summary.image("zym/vort", x_vort['zym'][:,::-1]),
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
        gen_list = self.batch_manager.random_list(self.b_num)
        x_xy = np.concatenate((gen_list['xym'],gen_list['xym_c']), axis=0)
        x_zy = np.concatenate((gen_list['zym'],gen_list['zym_c']), axis=0)
        save_image(x_xy, '{}/x_fixed_xym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)
        save_image(x_zy, '{}/x_fixed_zym_gt.png'.format(self.model_dir), padding=1, nrow=self.b_num)
        with open('{}/x_fixed_gt.txt'.format(self.model_dir), 'w') as f:
            f.write(str(gen_list['p']) + '\n')
            f.write(str(gen_list['z']))

        zi = np.zeros(shape=z_shape)            
        for i, z_gt in enumerate(gen_list['z']):
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
            self.G_s, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                      num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            self.G_ = curl(self.G_s)
        else:
            self.G_, _ = GeneratorBE3(self.z, self.filters, self.output_shape,
                                     num_conv=self.num_conv, repeat=self.repeat, reuse=True)        
    
    def generate(self, inputs, root_path=None, idx=None):
        # xy_list = []
        # zy_list = []
        xym_list = []
        zym_list = []

        # xyw_list = []
        # zyw_list = []
        xymw_list = []
        zymw_list = []

        for _, z_sample in enumerate(inputs):
            xym, zym = self.sess.run( # xy, zy, 
                [self.G['xym'], self.G['zym']], {self.y: z_sample}) # self.G['xy'], self.G['zy'], 
            # xy_list.append(xy)
            # zy_list.append(zy)
            xym_list.append(xym)
            zym_list.append(zym)

            xym, zym = self.sess.run( # xy, zy, 
                [self.G_vort['xym'], self.G_vort['zym']], {self.y: z_sample}) # self.G_vort['xy'], self.G_vort['zy'], 
            # xyw_list.append(xy)
            # zyw_list.append(zy)
            xymw_list.append(xym)
            zymw_list.append(zym)

        xym_list = xym_list[:-1] + xymw_list[:-1] + [xym_list[-1]] + [xymw_list[-1]]
        zym_list = zym_list[:-1] + zymw_list[:-1] + [zym_list[-1]] + [zymw_list[-1]]

        for tag, generated in zip(['xym','zym'], # '0_xy','0_zy',
                                  [xym_list, zym_list]): # xy_list, zy_list, 
            c_concat = np.concatenate(tuple(generated[:-2]), axis=0)
            c_path = os.path.join(root_path, '{}_{}.png'.format(idx,tag))
            save_image(c_concat, c_path, nrow=self.b_num, padding=1)
            print("[*] Samples saved: {}".format(c_path))

        gen_random = np.concatenate(tuple(xym_list[-2:]), axis=0)
        x_xy_path = os.path.join(root_path, 'x_fixed_xym_{}.png'.format(idx))
        save_image(gen_random, x_xy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_xy_path))

        gen_random = np.concatenate(tuple(zym_list[-2:]), axis=0)
        x_zy_path = os.path.join(root_path, 'x_fixed_zym_{}.png'.format(idx))
        save_image(gen_random, x_zy_path, nrow=self.b_num, padding=1)
        print("[*] Samples saved: {}".format(x_zy_path))
        

    def build_model_ae(self):
        if self.use_c:
            self.s, self.z, self.var = AE3(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                          num_conv=self.num_conv, repeat=self.repeat)
            _, self.x_ = jacobian3(self.s)
        else:
            self.x_, self.z, self.var = AE3(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                            num_conv=self.num_conv, repeat=self.repeat)
        self.x_img = denorm_img3(self.x_) # for debug

        self.x_jaco_, self.x_vort_ = jacobian3(self.x_)
        self.x_vort_ = denorm_img3(self.x_vort_)

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
            tf.summary.image("x/xym", self.x_img['xym'][:,::-1]),
            tf.summary.image("x/zym", self.x_img['zym'][:,::-1]),

            tf.summary.image("x/vort_xym", self.x_vort_['xym'][:,::-1]),
            tf.summary.image("x/vort_zym", self.x_vort_['zym'][:,::-1]),
            
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
        sample = self.batch_manager.random_list(self.b_num)    
        save_image(sample['xym'], '{}/xym_gt.png'.format(self.model_dir), nrow=self.b_num)
        save_image(sample['zym'], '{}/zym_gt.png'.format(self.model_dir), nrow=self.b_num)
        with open('{}/x_fixed_gt.txt'.format(self.model_dir), 'w') as f:
            f.write(str(sample['p']))
            f.write(str(sample['z']))
        
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
                self.autoencode(sample['x'], self.model_dir, idx=step)

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
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.test_b_num, self.res_z, self.res_y, self.res_x, 3])
        if self.use_c:
            self.s, self.z, self.var = AE3(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                          num_conv=self.num_conv, repeat=self.repeat, reuse=True)
            _, self.x_ = jacobian3(self.s)
        else:
            self.x_, self.z, self.var = AE3(self.x, self.filters, self.z_num, use_sparse=self.use_sparse,
                                              num_conv=self.num_conv, repeat=self.repeat, reuse=True)
        self.x_img = denorm_img3(self.x_) # for debug

    def autoencode(self, x, root_path=None, idx=None):
        # only for 2d
        x_img = self.sess.run(self.x_img, {self.x: x})
        xym = x_img['xym']
        x_path = os.path.join(root_path, 'xym_{}.png'.format(idx))
        save_image(xym, x_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(x_path))
        zym = x_img['zym']
        x_path = os.path.join(root_path, 'zym_{}.png'.format(idx))
        save_image(zym, x_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(x_path))