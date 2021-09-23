import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from crowd_nav.common.naive_transformer import TransformerEncoder

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            # layers.append(nn.BatchNorm1d(mlp_dims[i + 1]))
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()
        self.convz = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.convr = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.convq = nn.Linear(hidden_dim + input_dim, hidden_dim)

    def forward(self, x, h):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


# 和这里示意图一样，只不过h` 变成了 q  https://zhuanlan.zhihu.com/p/32481747
class GRUEXND(nn.Module):
    def __init__(self, input_dim, hidden_dims, last_relu=True):
        super(GRUEXND, self).__init__()
        self.convz = mlp(input_dim + hidden_dims[-1], hidden_dims, last_relu=last_relu)
        self.convr = mlp(input_dim + hidden_dims[-1], hidden_dims, last_relu=last_relu)
        self.convq = nn.Linear(hidden_dims[-1] + input_dim, hidden_dims[-1])

    # 怎么感觉这个是并行的
    def forward(self, x, h):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class GRUModel(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        self.grucell = nn.GRUCell(input_num, hidden_num)
        self.out_linear = nn.Linear(hidden_num, output_num)

    def forward(self, x, hid):
        if hid is None:
            hid = torch.randn(x.shape[0], self.hidden_size)
        next_hid = self.grucell(x, hid)  # 需要传入隐藏层状态
        y = self.out_linear(next_hid)
        return y, next_hid.detach()  # detach()和detach_()都可以使用


class NaiveGRUModel(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(NaiveGRUModel, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        # self.hidden = None

    def forward(self, state):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, hidden = self.GRU_layer(state)
        x = self.output_linear(x)
        x = x.view(-1, x.shape[2])

        return x


class ATCBasic(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dims, max_ponder=3, epsilon=0.05, last_relu=True):
        super(ATCBasic, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dims[-1]
        self.epsilon = epsilon
        self.rnn_cell = GRUEXND(input_dim, rnn_hidden_dims, last_relu)

        self.max_ponder = max_ponder
        self.ponder_linear = nn.Linear(rnn_hidden_dims[-1], 1)

    def forward(self, input, hx=None):
        # Pre-allocate variables
        # time_size, batch_size, input_dim_size = input_.size()
        size = input.shape
        input_ = input.view(-1, size[2])
        selector = input_.data.new(input_.shape[0]).byte()
        ponder_times = []
        accum_p = 0
        accum_hx = torch.zeros([input_.shape[0], self.rnn_hidden_dim]).cuda()
        step_count = 0
        self.step_cnt = 0
        self.step2_cnt = 0
        self.step3_cnt = 0
        # For each t
        for act_step in range(self.max_ponder):
            hx = self.rnn_cell(input_, hx)
            hx = hx.view(-1, self.rnn_hidden_dim)
            halten_p = torch.sigmoid(self.ponder_linear(hx))  # halten state
            accum_p += halten_p
            accum_hx += halten_p * hx
            step_count += 1
            selector = (accum_p < 1 - self.epsilon).data
            if not selector.any():  # selector has no true elements
                break
        
        if step_count == 1:
            self.step_cnt = step_count
        elif step_count == 2:
            self.step2_cnt = step_count
        else:
            self.step3_cnt = step_count

        # ponder_times.append(step_count.data.cpu().numpy())
        # accum_hx = accum_hx.view(size[0], size[1], -1)
        # hx = torch.mean(accum_hx, 1, True) #[B, C]
        hx = accum_hx / step_count
        env_score = accum_p / step_count
        return hx, env_score


class ATCBasicTfencoder(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dims, max_ponder=3, epsilon=0.05, last_relu=True):
        super(ATCBasicTfencoder, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dims[-1]
        self.epsilon = epsilon
        self.rnn_cell = GRUEXND(input_dim, rnn_hidden_dims, last_relu)
        self.transformer = TransformerEncoder(source_dims=13, k_dims=16, v_dims=16, n_heads=3, layer_cnt=1)
        self.transition_layer = nn.Linear(13, rnn_hidden_dims[-1])

        self.max_ponder = max_ponder
        self.ponder_linear = nn.Linear(rnn_hidden_dims[-1], 1)

    def forward(self, input, hx=None):
        # Pre-allocate variables
        # time_size, batch_size, input_dim_size = input_.size()
        size = input.shape
        input_ = input.view(-1, size[2])
        selector = input_.data.new(input_.shape[0]).byte()
        ponder_times = []
        accum_p = 0
        accum_hx = torch.zeros([input_.shape[0], self.rnn_hidden_dim]).cuda()
        step_count = 0
        # For each t
        for act_step in range(self.max_ponder):
            hx = self.transformer(input)
            hx = self.transition_layer(hx)
            hx = hx.view(-1, self.rnn_hidden_dim)
            halten_p = torch.sigmoid(self.ponder_linear(hx))  # halten state
            accum_p += halten_p
            accum_hx += halten_p * hx
            step_count += 1
            selector = (accum_p < 1 - self.epsilon).data
            if not selector.any():  # selector has no true elements
                break

        # ponder_times.append(step_count.data.cpu().numpy())
        # accum_hx = accum_hx.view(size[0], size[1], -1)
        # hx = torch.mean(accum_hx, 1, True) #[B, C]
        hx = accum_hx / step_count
        # env_score = accum_p / step_count
        return hx, step_count