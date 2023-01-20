import time

from models import DLinear, NLinear, Linear, DSTLinear
from torchstat import stat
import torch


class CMD(object):
    def __init__(self, enc_in, seq_len, pred_len):
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_zoo() -> dict:
    arg = CMD(862, 96, 720)
    dstlinear = DSTLinear.Model(arg)
    nlinear = NLinear.Model(arg)
    dlinear = DLinear.Model(arg)
    linear = Linear.Model(arg)
    model_zoo = {"DSTLinear": dstlinear, "NLinear": nlinear, "DLinear": dlinear, "Linear": linear}
    return model_zoo


def param_num(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("模型参数量：%.2fK" % (total / 1e3))


def inference_time(model: torch.nn.Module):
    x = torch.randn(16, 96, 862).to(device)  # (batch, L, dim)
    model = model.to(device)
    t = []
    for i in range(10):
        s = time.time()
        model(x)
        e = time.time()
        t.append(e - s)
    print("模型平均推理用时：{}ms".format(1000 * sum(t) / len(t)))


if __name__ == '__main__':
    model_zoo = get_zoo()
    for name, model in model_zoo.items():
        print("\n-------------------------{}-------------------------".format(name))
        param_num(model)
        inference_time(model)
