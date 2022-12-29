import argparse
import os

import torch
import random
import numpy as np

from run import main_etth1, main_etth2, main_ettm1, main_ettm2, main_ill, main_electricity, main_exchange_rate, \
    main_traffic, main_weather

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

data_dict = {
    1: main_etth1.MainETTh1,
    2: main_etth2.MainETTh2,
    3: main_ettm1.MainETTm1,
    4: main_ettm2.MainETTm2,
    5: main_electricity.MainElectricity,
    6: main_exchange_rate.MainExchangeRate,
    7: main_traffic.MainTraffic,
    8: main_weather.MainWeather,
    9: main_ill.MainIll
}


def train():
    parser = argparse.ArgumentParser(description='choose a dataset')
    parser.add_argument('--dataset_no', type=int, default=1, help='dataset_no')
    args = parser.parse_args()

    data_dict[args.dataset_no]().main()


if __name__ == '__main__':
    train()
