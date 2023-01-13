import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels: list, kernel_size=2, dropout=0.4):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class MLP(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.4)

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        # hidden = self.fc2(self.act(self.fc1(input_data)))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class Model(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # attributes
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layer = 1

        self.temporal_emb_dim = self.enc_in
        self.spatial_emb_dim = self.enc_in
        self.embed_dim = self.enc_in

        # spatial embeddings
        self.spatial_emb = nn.Linear(self.enc_in, self.spatial_emb_dim)  # enc_in->spatial_emb_dim

        # temporal embeddings
        self.temporal_emb = nn.Linear(4, self.temporal_emb_dim)  # 4->temporal_emb_dim

        # TCN layer
        self.Front_TCN = TemporalConvNet(self.enc_in, [self.embed_dim])

        # encoding embed_dim + temporal_emb_dim + temp_dim_diw + spatial_emb_dim
        self.hidden_dim = self.enc_in
        block = [MLP(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        block += [nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.enc_in, kernel_size=1)]
        self.encoder = nn.Sequential(*block)

        # regression
        self.regression_layer = nn.Linear(self.seq_len, self.pred_len)

        self.initialize_weights()

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        """Feed forward of STLinear.

        Args:
            x (torch.Tensor): history data with shape [B, L, dim]
            x_mark (torch.Tensor):
                history data timestamp with shape [B, L, dim]
                which:dim -> (HoursOfDay, DayOfWeek, DayOfMonth, DayOfYear)

        Returns:
            torch.Tensor: prediction wit shape [B, H, dim]
        """

        # seq_last = x[:, -1:, :].detach()
        # x = x - seq_last

        # time series embedding: TCN
        batch_size, _, dim = x.shape

        # enc_in->spatial_emb_dim
        # node_emb = self.spatial_emb(x).permute(0, 2, 1)  # [batch, spatial_emb_dim, len]
        # temporal embeddings
        # tem_emb = self.temporal_emb(x_mark).permute(0, 2, 1)  # [batch, temporal_emb_dim, len]

        # concate all embeddings     hidden: [batch, hidden_dim, len, 1]

        # encoding   # hidden: [batch, enc_in, len]
        # hidden = self.encoder(x_residual + node_emb + tem_emb)

        x_residual = self.Front_TCN(x.permute(0, 2, 1))  # [batch, embed_dim, len]
        # regression
        # output = self.regression_layer(x.permute(0, 2, 1) + node_emb + tem_emb).permute(0, 2, 1)
        output = self.regression_layer(x.permute(0, 2, 1) + x_residual).permute(0, 2, 1)
        # output = output + seq_last
        return output


if __name__ == '__main__':
    class cmd(object):
        def __init__(self, enc_in, seq_len, pred_len):
            self.enc_in = enc_in
            self.seq_len = seq_len
            self.pred_len = pred_len


    # (32,104,7)

    x = torch.randn(32, 104, 7)
    x_mark = torch.randn(32, 104, 4)
    arg = cmd(7, 104, 24)
    model = Model(arg)

    y = model(x, x_mark)  # (32,24,7)
    print(y.shape)
