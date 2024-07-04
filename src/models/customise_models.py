import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, feature_per_shell, num_shells, n_mix, act_fn):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, feature_per_shell, kernel_size=(n_mix, feature_per_shell),
            stride=1, padding=(n_mix-1 ,0), dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(feature_per_shell)
        self.conv2 = nn.Conv2d(1, feature_per_shell, kernel_size=(n_mix, feature_per_shell),
            stride=1, padding=(n_mix-1 ,0), dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(feature_per_shell)
        self.act_fn = act_fn
        self.selectshell = torch.arange(num_shells)

    def forward(self, x):
        device = x.device
        identity = x # (N, 1, 10, 80)
        h = self.conv1(x) # (N, 80, 11, 1)
        h = self.bn1(h) # norm by the same feature
        h = self.act_fn(h)
        h = h.permute(0,3,2,1) # (N, 1, 11, 80)
        h = h.index_select(2, self.selectshell.to(device)) # (N, 1, 10, 80)
        h = self.conv2(h) # (N, 80, 11, 1)
        h = self.bn2(h)
        h = h.permute(0,3,2,1) # (N, 1, 11, 80)
        h = h.index_select(2, self.selectshell.to(device)) # (N, 1, 10, 80)
        h += identity
        h = self.act_fn(h)
        return h


class FeatIntResNet(nn.Module):
    def __init__(self, feature_per_shell=80, num_shells=6, n_blocks=5, \
            n_mix=2, n_class=2, activation='relu', noise=None):
        super(FeatIntResNet, self).__init__()
        self.feature_per_shell = feature_per_shell
        self.num_shells = num_shells
        self.n_blocks = n_blocks
        self.act_fn = getattr(nn.functional, activation)
        self.noise = noise
        blocks = []
        for _ in range(self.n_blocks):
            resblock = ResBlock(feature_per_shell=self.feature_per_shell, num_shells=self.num_shells, \
                n_mix=n_mix, act_fn=self.act_fn)
            blocks.append(resblock)
        self.norm_layer = nn.BatchNorm2d(self.feature_per_shell)
        self.blocks = nn.Sequential(*blocks)
        # self.fc_layer = nn.Linear(self.num_shells*self.feature_per_shell, n_class)
        self.fc_layer = nn.Linear(self.feature_per_shell, n_class)
    
    def forward(self, x):
        # reshape layer: (N, C, H, W) = (N, 1, 10, 80)
        h = x.view((-1, 1, self.num_shells, self.feature_per_shell))
        h = h.permute(0, 3, 2, 1) # (N, 80, 10, 1)
        # normalization layer
        h = self.norm_layer(h)
        h = h.permute(0, 3, 2, 1) # (N, 1, 10, 80)
        if self.training and self.noise:
            h = h + torch.randn_like(h) * self.noise
        h = self.blocks(h)
        # h = nn.Flatten(start_dim=1, end_dim=-1)(h) # (N, 800)
        h = nn.AvgPool2d(kernel_size=(self.num_shells, 1))(h) # (N, 80)
        h = h.view(-1, self.feature_per_shell)
        h = self.fc_layer(h)
        return h


def get_fnn_model(in_dim, h_dims, n_class, dropout=0.5, bias=True):
    assert type(h_dims) == list
    hidden_layes = []
    norm = nn.BatchNorm1d(in_dim)
    if len(h_dims) != 0:
        i2h = nn.Linear(in_dim, h_dims[0], bias=bias)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout)
        for i in range(len(h_dims)-1):
            h2h = nn.Linear(h_dims[i], h_dims[i+1], bias=bias)
            hidden_layes.extend([h2h, dropout, relu])
        h2o = nn.Linear(h_dims[-1], n_class, bias=bias)
        model = nn.Sequential(norm, i2h, dropout, relu, *hidden_layes, h2o)
    else:
        i2h = nn.Linear(in_dim, n_class, bias=bias)
        model = nn.Sequential(norm, i2h)
    return model
