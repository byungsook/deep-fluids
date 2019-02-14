import numpy as np
import tensorflow as tf

from config import get_config
from util import prepare_dirs_and_logger, save_config

from trainer import Trainer
from trainer3 import Trainer3

def main(config):
    prepare_dirs_and_logger(config)
    tf.set_random_seed(config.random_seed)

    if 'nn' in config.arch:
        from data_nn import BatchManager
    else:
        from data import BatchManager
    batch_manager = BatchManager(config)

    if config.is_3d:
        trainer = Trainer3(config, batch_manager)
    else:
        trainer = Trainer(config, batch_manager)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
