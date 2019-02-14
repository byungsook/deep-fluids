from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)
   
def conv2d(x, o_dim, data_format='NHWC', name=None, k=4, s=2, act=None):
    return slim.conv2d(x, o_dim, k, stride=s, activation_fn=act, scope=name, data_format=data_format)

def conv3d(x, o_dim, data_format='NDHWC', name=None, k=4, s=2, act=None):
    return slim.conv3d(x, o_dim, k, stride=s, activation_fn=act, scope=name, data_format=data_format)
    # return tf.layers.conv3d(x, o_dim, k, (s,s,s), 'SAME', activation=act, name=name,
    #                         kernel_initializer=tf.contrib.layers.xavier_initializer())

def deconv2d(x, o_dim, data_format='NHWC', name=None, k=4, s=2, act=None):
    return slim.conv2d_transpose(x, o_dim, k, stride=s, activation_fn=act, scope=name, data_format=data_format)

def linear(x, o_dim, name=None, act=None):
    return slim.fully_connected(x, o_dim, activation_fn=act, scope=name)

def batch_norm(x, train, data_format='NHWC', name=None, act=lrelu, epsilon=1e-5, momentum=0.9):
    return slim.batch_norm(x,
                        decay=momentum,
                        updates_collections=None,
                        epsilon=epsilon,
                        scale=True,
                        fused=True,
                        is_training=train,
                        activation_fn=act,
                        data_format=data_format,
                        scope=name)

def inst_norm(x, train, data_format='NHWC', name=None, affine=False, act=lrelu, epsilon=1e-5):
    with tf.variable_scope(name, default_name='Inst', reuse=None) as vs:
        if x.get_shape().ndims == 4 and data_format == 'NCHW':
            x = nchw_to_nhwc(x)

        if x.get_shape().ndims == 4:
            mean_dim = [1,2]
        else: # 2
            mean_dim = [1]

        mu, sigma_sq = tf.nn.moments(x, mean_dim, keep_dims=True)
        inv = tf.rsqrt(sigma_sq+epsilon)
        normalized = (x-mu)*inv

        if affine:
            var_shape = [x.get_shape()[-1]]
            shift = slim.model_variable('shift', shape=var_shape, initializer=tf.zeros_initializer)
            scale = slim.model_variable('scale', shape=var_shape, initializer=tf.ones_initializer)
            out = scale*normalized + shift
        else:
            out = normalized
        
        if x.get_shape().ndims == 4 and data_format == 'NCHW':
            out = nhwc_to_nchw(out)

        if act is None: return out
        else: return act(out)

def resize_nearest_neighbor(x, new_size, data_format='NHWC'):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format='NHWC'):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

def upscale3(x, scale):
    b, d, h, w, c = int_shape(x)

    hw = tf.reshape(tf.transpose(x, [0,2,3,1,4]), [b,h,w,d*c])
    h *= scale
    w *= scale
    hw = tf.image.resize_nearest_neighbor(hw, (h,w))
    hw = tf.reshape(hw, [b,h,w,d,c])

    dh = tf.reshape(tf.transpose(hw, [0,3,1,2,4]), [b,d,h,w*c])
    d *= scale    
    dh = tf.image.resize_nearest_neighbor(dh, (d,h))
    return tf.reshape(dh, [b,d,h,w,c])

def var_on_cpu(name, shape, initializer, dtype=tf.float32):
    return slim.model_variable(name, shape, dtype=dtype, initializer=initializer, device='/CPU:0')

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format='NHWC'):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format='NHCW'):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1,2,3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def to_nhwc_numpy(image):
    if image.shape[1] in [1,2,3]:
        new_image = image.transpose([0, 2, 3, 1])
    else:
        new_image = image
    return new_image

def add_channels(x, num_ch=1, data_format='NHWC'):
    b, h, w, c = get_conv_shape(x, data_format)
    if data_format == 'NCHW':
        x = tf.concat([x, tf.zeros([b, num_ch, h, w])], axis=1)
    else:
        x = tf.concat([x, tf.zeros([b, h, w, num_ch])], axis=-1)
    return x

def remove_channels(x, data_format='NHWC'):
    b, h, w, c = get_conv_shape(x, data_format)
    if data_format == 'NCHW':
        x, _ = tf.split(x, [3, -1], axis=1)
    else:
        x, _ = tf.split(x, [3, -1], axis=3)
    return x

def denorm_img(norm, data_format='NHWC'):
    _, _, _, c = get_conv_shape(norm, data_format)
    if c == 2:
        norm = add_channels(norm, num_ch=1, data_format=data_format)
    elif c > 3:
        norm = remove_channels(norm, data_format=data_format)
    img = tf.cast(tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255), tf.uint8)
    return img

def plane_view(x, xy_plane=True, project=True):
    x_shape = int_shape(x) # bzyxd
    c_id = [int(x_shape[1]/2), int(x_shape[3]/2)]
    
    if xy_plane:
        if project:
            x = tf.reduce_mean(x, 1)
        else:
            x = tf.squeeze(tf.slice(x, [0,c_id[0],0,0,0], [-1,1,-1,-1,-1]), [1])            
    else:
        if project:
            x = tf.transpose(tf.reduce_mean(x, 3), [0,2,1,3])
        else:
            x = tf.transpose(tf.squeeze(
                    tf.slice(x, [0,0,0,c_id[1],0], [-1,-1,-1,1,-1]),
                    [3]), [0,2,1,3])

    x = tf.cast(tf.clip_by_value((x + 1)*127.5, 0, 255), tf.uint8)
    return x

def denorm_img3(x):
    xy = plane_view(x, xy_plane=True, project=True)
    zy = plane_view(x, xy_plane=False, project=True)
    xym = plane_view(x, xy_plane=True, project=False)
    zym = plane_view(x, xy_plane=False, project=False)
    return {'xy': xy, 'zy': zy, 'xym': xym, 'zym': zym}

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def reshape(x, h, w, c, data_format='NHWC'):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def jacobian(x, data_format='NHCW'):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)

    dudx = x[:,:,1:,0] - x[:,:,:-1,0]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dvdy = x[:,1:,:,1] - x[:,:-1,:,1]
    
    dudx = tf.concat([dudx,tf.expand_dims(dudx[:,:,-1], axis=2)], axis=2)
    dvdx = tf.concat([dvdx,tf.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = tf.concat([dudy,tf.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    dvdy = tf.concat([dvdy,tf.expand_dims(dvdy[:,-1,:], axis=1)], axis=1)

    j = tf.stack([dudx,dudy,dvdx,dvdy], axis=-1)
    w = tf.expand_dims(dvdx - dudy, axis=-1) # vorticity (for visualization)

    if data_format == 'NCHW':
        j = nhwc_to_nchw(j)
        w = nhwc_to_nchw(w)
    return j, w

def jacobian3(x):
    # x: bzyxd
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = tf.concat((dudx, tf.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = tf.concat((dvdx, tf.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = tf.concat((dwdx, tf.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = tf.concat((dudy, tf.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = tf.concat((dvdy, tf.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = tf.concat((dwdy, tf.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = tf.concat((dudz, tf.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = tf.concat((dvdz, tf.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = tf.concat((dwdz, tf.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = tf.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = tf.stack([u,v,w], axis=-1)
    
    return j, c
    
def curl(x, data_format='NHWC'):
    if data_format == 'NCHW': x = nchw_to_nhwc(x)

    u = x[:,1:,:,0] - x[:,:-1,:,0] # ds/dy
    v = x[:,:,:-1,0] - x[:,:,1:,0] # -ds/dx,
    u = tf.concat([u, tf.expand_dims(u[:,-1,:], axis=1)], axis=1)
    v = tf.concat([v, tf.expand_dims(v[:,:,-1], axis=2)], axis=2)
    c = tf.stack([u,v], axis=-1)

    if data_format == 'NCHW': c = nhwc_to_nchw(c)
    return c
    
def divergence(x, data_format='NHWC'):
    if data_format == 'NCHW': x = nchw_to_nhwc(x)

    dudx = x[:,:-1,1:,0] - x[:,:-1,:-1,0]
    dvdy = x[:,1:,:-1,1] - x[:,:-1,:-1,1]
    div = tf.expand_dims(dudx + dvdy, axis=-1)

    if data_format == 'NCHW': div = nhwc_to_nchw(div)
    return div

def divergence3(x):
    dudx = x[:,:-1,:-1,1:,0] - x[:,:-1,:-1,:-1,0]
    dvdy = x[:,:-1,1:,:-1,1] - x[:,:-1,:-1,:-1,1]
    dwdz = x[:,1:,:-1,:-1,2] - x[:,:-1,:-1,:-1,2]
    return tf.expand_dims(dudx + dvdy + dwdz, axis=-1)

def pgrad(x, data_format):
    # pressure gradient
    if data_format == 'NCHW': x = nchw_to_nhwc(x)

    u = x[:,:,1:,0] - x[:,:,:-1,0] # dp/dx,
    v = x[:,1:,:,0] - x[:,:-1,:,0] # dp/dy
    u = tf.concat([u,tf.expand_dims(u[:,:,-1], axis=2)], axis=2)
    v = tf.concat([v,tf.expand_dims(v[:,-1,:], axis=1)], axis=1)
    g = tf.stack([u,v], axis=-1)

    if data_format == 'NCHW': g = nhwc_to_nchw(g)
    return g

def vort_np(x):
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = np.concatenate([dvdx,np.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = np.concatenate([dudy,np.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    return np.expand_dims(dvdx - dudy, axis=-1)

def curl_np(x):
    u = x[:,1:,:,0] - x[:,:-1,:,0] # ds/dy
    u = np.concatenate([u,np.expand_dims(u[:,-1,:], axis=1)], axis=1)
    v = x[:,:,:-1,0] - x[:,:,1:,0] # -ds/dx
    v = np.concatenate([v,np.expand_dims(v[:,:,-1], axis=2)], axis=2)
    return np.stack([u,v], axis=-1)

def grad_np(x):
    u = x[:,:,1:,0] - x[:,:,:-1,0] # dp/dx,
    v = x[:,1:,:,0] - x[:,:-1,:,0] # dp/dy
    u = np.concatenate([u,np.expand_dims(u[:,:,-1], axis=2)], axis=2)
    v = np.concatenate([v,np.expand_dims(v[:,-1,:], axis=1)], axis=1)
    return np.stack([u,v], axis=-1)

def plane_view_np(x, xy_plane=True, project=True):
    x_shape = x.shape # zyxd
    c_id = [int(x_shape[0]/2), int(x_shape[2]/2)]
    
    if xy_plane:
        if project:
            x = np.mean(x, axis=0)
        else:
            x = x[c_id[0],:,:,:]
    else:
        if project:
            x = np.mean(x, axis=2).transpose([1,0,2])
        else:
            x = x[:,:,c_id[1],:].transpose([1,0,2])

    x = np.clip((x+1)*127.5, 0, 255)
    return x

def jacobian_np3(x):
    # x: bzyxd
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = np.stack([u,v,w], axis=-1)
    return j, c

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
# https://github.com/tensorflow/models/blob/master/compression/image_encoder/msssim.py
def fspecial_gauss(size, sigma, channels):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size

    x = x.reshape(x.shape+(1,1))
    x = np.repeat(x, channels, axis=2)
    x = np.repeat(x, channels, axis=3)

    y = y.reshape(y.shape+(1,1))
    y = np.repeat(y, channels, axis=2)
    y = np.repeat(y, channels, axis=3)        

    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def ssim(img1, img2, mean_metric=True, 
         filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03,
         min_val=-1.0, max_val=1.0):
    
    # input should be rescaled to [-1,1]
    img_shape = img1.get_shape()
    height = img_shape[1].value
    width = img_shape[2].value
    channels = img_shape[3].value
    # print(img_shape)

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)
    # print(size)

    # Scale down sigma if a smaller filter size is used.
    sigma = filter_sigma * size / filter_size if filter_size else 0
    # print(sigma)

    # ! normalize image to [0,1]
    img1 = (img1 - min_val) / (max_val - min_val)
    img2 = (img2 - min_val) / (max_val - min_val)

    if filter_size:
        window = fspecial_gauss(size, sigma, channels) # window shape [size, size]
        mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
        sigma11 = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID')
        sigma22 = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID')
        sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID')
    else:
        mu1 = img1, mu2 = img2
        sigma11 = img1*img1
        sigma22 = img2*img2
        sigma12 = img1*img2
    
    mu11 = mu1*mu1
    mu22 = mu2*mu2
    mu12 = mu1*mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    L = 1.0 # max scale, already normalized to 1
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    value = ((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2) 
    if mean_metric: return tf.reduce_mean(value)

    result = {'ssim_map': value, 'cs_map': v1/v2, 'g': window}
    return result


def ms_ssim(img1, img2, mean_metric=True, min_val=-1.0, max_val=1.0):
    # input should be rescaled to [-1,1]
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    mssim = []
    mcs = []

    for w in weight:
        result = ssim(img1, img2, mean_metric=False, min_val=min_val, max_val=max_val)
        mssim.append(tf.reduce_mean(result['ssim_map']))
        mcs.append(tf.reduce_mean(result['cs_map']))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # ! doesn't work        
    # filter_sigmas = [0.5, 1, 2, 4, 8]
    # cs_map0 = None
    # for i, filter_sigma in enumerate(filter_sigmas):        
    #     result = ssim(img1, img2, filter_sigma=filter_sigma, 
    #                   min_val=min_val, mean_metric=False)

    #     if i == 0: cs_map0 = result['cs_map']

    #     mssim.append(tf.reduce_mean(result['ssim_map']))
    #     mcs.append(tf.reduce_mean(tf.nn.conv2d(cs_map0, result['g'], strides=[1,1,1,1], padding='VALID')))

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)
    level = len(weight)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric: value = tf.reduce_mean(value)
    return value


def main(_):
    from skimage import data, transform, img_as_float
    import matplotlib.pyplot as plt

    color = False
    if color:
        image = data.astronaut()
    else: # [h,w] -> [h,w,1]
        image = data.camera()
        image = np.expand_dims(image, axis=-1)

    # image = transform.resize(image, output_shape=[128, 128])

    img = img_as_float(image)
    print(img.shape)
    rows, cols, channels = img.shape

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    noise[np.random.random(size=noise.shape) > 0.5] *= -1

    img_noise = img + noise
    img_noise = np.clip(img_noise, a_min=0, a_max=1)

    plt.figure()
    plt.subplot(121)
    if color:
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(img_noise)
    else:
        plt.imshow(img[:,:,0], cmap='gray')
        plt.subplot(122)
        plt.imshow(img_noise[:,:,0], cmap='gray')
    plt.show()


    ## TF CALC START
    image1 = tf.placeholder(tf.float32, shape=[rows, cols, channels])
    image2 = tf.placeholder(tf.float32, shape=[rows, cols, channels])

    def image_to_4d(image):
        image = tf.expand_dims(image, 0)
        return image

    image4d_1 = image_to_4d(image1)
    image4d_2 = image_to_4d(image2)

    print(img.min(), img.max(), img_noise.min(), img_noise.max())
    ssim_index = ssim(image4d_1, image4d_2) #, min_val=0.0, max_val=1.0)
    msssim_index = ms_ssim(image4d_1, image4d_2) #, min_val=0.0, max_val=1.0)

    # img *= 255
    # img_noise *= 255
    # ssim_index = ssim(image4d_1, image4d_2, min_val=0.0, max_val=255.0)
    # msssim_index = ms_ssim(image4d_1, image4d_2, min_val=0.0, max_val=255.0)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf_ssim_none = sess.run(ssim_index,
                                feed_dict={image1: img, image2: img})
        tf_ssim_noise = sess.run(ssim_index,
                                feed_dict={image1: img, image2: img_noise})

        tf_msssim_none = sess.run(msssim_index,
                                feed_dict={image1: img, image2: img})
        tf_msssim_noise = sess.run(msssim_index,
                                feed_dict={image1: img, image2: img_noise})
    ###TF CALC END

    print('tf_ssim_none', tf_ssim_none)
    print('tf_ssim_noise', tf_ssim_noise)
    print('tf_msssim_none', tf_msssim_none)
    print('tf_msssim_noise', tf_msssim_noise)


if __name__ == '__main__':    
    tf.app.run()