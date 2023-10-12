import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F



class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            input_size=172,
            hidden_size=512,
            output_size=512,
            bidirectional=True,
            use_residual=False,
            feed_forward=True,
            dropout=0.5,
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )

        self.linear = None

        if bidirectional:
            hidden_size = hidden_size * 2

        if use_residual:
            self.linear = nn.Linear(hidden_size, input_size)
        elif feed_forward:
            self.linear = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=0.5)

        self.use_residual = use_residual

    def forward(self, x, init_state=None):
        n, t, f = x.shape
        # x = x.permute(1,0,2) # NTF -> TNF

        h, _ = self.gru(x, init_state)
        if self.linear:
            #y = self.norm(h)
            y = F.elu(h)

            y = self.linear(y.contiguous().view(-1, y.shape[-1]))
            #y = self.norm(y)

            y = y.view(n, t, y.shape[-1])
        if self.use_residual and y.shape[-1] == x.shape[-1]:
            y = y + x

        # y = y.permute(1,0,2) # TNF -> NTF
        return y

class Tracker(nn.Module):
    def __init__(
            self,
            input_size=172,
            hidden_size=512,
            output_size=512,
    ):
        super(Tracker, self).__init__()
        self.encoder1 = TemporalEncoder(
            n_layers=1,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            bidirectional=False,
            use_residual=False,
            feed_forward=True,
        )

        self.encoder2 = TemporalEncoder(
            n_layers=1,
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            bidirectional=False,
            use_residual=True,
            feed_forward=True,
        )

        self.encoder3 = TemporalEncoder(
            n_layers=1,
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            bidirectional=False,
            use_residual=False,
            feed_forward=True,
        )


        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x, init_state=None):
        n, t, f = x.shape
        # x = x.permute(1,0,2) # NTF -> TNF

        h = self.encoder1(x, init_state)
        h = self.dropout1(h)
        h = self.encoder2(h, init_state)
        h = self.dropout2(h)
        y = self.encoder3(h, init_state)

        # y = y.permute(1,0,2) # TNF -> NTF
        return y




class HybriNet(nn.Module):
    def __init__(
            self,
            n_layers=3,
            input_size=60 + 66 + 69,
            hidden_size=1024,
            bidirectional=True,
    ):
        super(HybriNet, self).__init__()
        self.n_layers = n_layers

        self.limb_tracker = Tracker(
            input_size=56+63,
            hidden_size=hidden_size,
            output_size=24,

        )

        self.joint_tracker = Tracker(
            input_size=56+63+24,
            hidden_size=hidden_size,
            output_size=63,

        )

        self.hybrik = Tracker(
            input_size=56+63+63,
            hidden_size=hidden_size,
            output_size=24*6,
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

#         init_state = self.state_encoder(init_info)
#         init_state = init_state.reshape(batch_size, self.n_layers, -1)
#         init_state = init_state.permute([1, 0, 2]).contiguous()

        p_limb = self.limb_tracker(x)
        x1 = torch.cat([x, p_limb], 2)

        p_body = self.joint_tracker(x1)
        x2 = torch.cat([x, p_body], 2)

        fullpose_6d = self.hybrik(x2)
        
        data = {}
        data["p_limb"] = p_limb
        data["p_body"] = p_body
        data["fullpose_6d"] = fullpose_6d
        
        return data


class SimpleNet(nn.Module):
    def __init__(
            self,
            n_layers=2,
            input_size=60 + 66 + 69,
            hidden_size=512,
            bidirectional=True,
    ):
        super(SimpleNet, self).__init__()
        self.n_layers = n_layers

        self.state_encoder = nn.Linear(60 + 66 + 75, hidden_size * n_layers)


        self.hybrik = TemporalEncoder(
            n_layers=n_layers,
            input_size=60 + 63 + 69,
            hidden_size=hidden_size,
            output_size=72,
            bidirectional=False,
            use_residual=False,
            feed_forward=True,
        )

    def forward(self, x, init_info):
        batch_size, seq_len, dim = x.shape

        # init_state = self.state_encoder(init_info)
        # init_state = init_state.reshape(batch_size, self.n_layers, -1)
        # init_state = init_state.permute([1, 0, 2]).contiguous()


        transpose = self.hybrik(x)

        return transpose
