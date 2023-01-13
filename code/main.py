import torch
import random
import numpy as np

import main_etth1
import main_ettm1
import main_exchange_rate
import main_weather
import main_traffic
import main_etth2
import main_ettm2
import main_ill
import main_electricity

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


def train(i: int):
    print(data_dict[i].__name__)
    data_dict[i]().main()


if __name__ == '__main__':
    i = 9
    train(i)
