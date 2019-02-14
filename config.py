import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--is_3d', type=str2bool, default=False)
net_arg.add_argument('--res_x', type=int, default=96)
net_arg.add_argument('--res_y', type=int, default=128)
net_arg.add_argument('--res_z', type=int, default=32)
net_arg.add_argument('--repeat', type=int, default=0)
net_arg.add_argument('--filters', type=int, default=128)
net_arg.add_argument('--num_conv', type=int, default=4)
net_arg.add_argument('--use_curl', type=str2bool, default=True)
net_arg.add_argument('--w1', type=float, default=1.0, help='weight for l1')
net_arg.add_argument('--w2', type=float, default=1.0, help='weight for jacobian')
net_arg.add_argument('--w3', type=float, default=0.005, help='weight for discriminator')
net_arg.add_argument('--arch', type=str, default='de', choices=['de', 'dg', 'ae', 'nn'],
                     help='dec, dec+discriminator, auto-encoder, multi-layer perceptron')
# for AE and NN
net_arg.add_argument('--z_num', type=int, default=16)
net_arg.add_argument('--use_sparse', type=str2bool, default=False)
net_arg.add_argument('--sparsity', type=float, default=0.01)
net_arg.add_argument('--w4', type=float, default=1.0, help='weight for p')
net_arg.add_argument('--w5', type=float, default=1.0, help='weight for sparsity constraint')
net_arg.add_argument('--w_size', type=int, default=5)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='smoke_pos21_size5_f200')
data_arg.add_argument('--batch_size', type=int, default=8)
data_arg.add_argument('--test_batch_size', type=int, default=100)
data_arg.add_argument('--num_worker', type=int, default=2)
data_arg.add_argument('--data_type', type=str, default='velocity')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--start_step', type=int, default=0)
train_arg.add_argument('--max_epoch', type=int, default=100)
train_arg.add_argument('--lr_update_step', type=int, default=120000)
train_arg.add_argument('--lr_max', type=float, default=0.0001)
train_arg.add_argument('--lr_min', type=float, default=0.0000025)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--lr_update', type=str, default='decay',
                       choices=['decay', 'step'])

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_dir', type=str, default='log')
misc_arg.add_argument('--tag', type=str, default='tag')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--code_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=500)
misc_arg.add_argument('--test_step', type=int, default=1000)
misc_arg.add_argument('--save_sec', type=int, default=3600)
misc_arg.add_argument('--random_seed', type=int, default=123)
misc_arg.add_argument('--gpu_id', type=str, default='0')

def get_config():
    config, unparsed = parser.parse_known_args()
    
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id # "0, 1" for multiple

    return config, unparsed