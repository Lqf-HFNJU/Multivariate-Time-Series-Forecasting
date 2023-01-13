import argparse
import os

import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

from run.main_basic import MainBasic

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class MainETTm1(MainBasic):

    def init_param(self):
        self.H = [96, 192, 336, 720]
        self.batch_size = [8, 8, 8, 8]
        self.lr = 0.0001

    def start(self, args, batch_size: int, pred_len: int, lr: float):
        self.init_path()
        seq_len = 336
        model_name = 'TCNLinear'

        """
            >logs/LongForecasting/$model_name'_'ETTh1_$seq_len'_'96.log 
        """

        args.is_training = 1
        args.root_path = './dataset/'
        args.data_path = 'ETTm1.csv'
        args.model_id = 'ETTm1_' + str(seq_len) + '_' + str(pred_len)
        args.data = 'ETTm1'
        args.features = 'M'
        args.seq_len = seq_len
        args.pred_len = pred_len
        args.enc_in = 7
        args.des = 'Exp'
        args.itr = 1
        args.batch_size = batch_size
        args.learning_rate = lr
        return args


if __name__ == '__main__':
    MainETTm1().main()
