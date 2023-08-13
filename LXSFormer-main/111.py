import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import torch
import torch.nn.functional as F
from basicsr.archs import SwinT

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    return nn.Sequential(#400epoch 32.726 28.623
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1),
                           bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), (3, 1), 1, (1, 0), bias=True),
        nn.Conv2d(int(in_channels * 0.5), out_channels, 1, 1, bias=True)
    )

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class HBCT(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(HBCT, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.esa = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        # self.sparatt = Spartial_Attention.Spartial_Attention()
        self.swinT = SwinT.SwinT()

    def forward(self, input):
        input = self.esa2(input)
        input = self.swinT(input)
        out_fused = self.esa(self.c1_r(input))
        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)



##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# pytorch中torch.chunk()使用方法
# chunk方法可以对张量分块，返回一个张量列表：
# torch.chunk(tensor, chunks, dim=0) → List of Tensors
# tensor (Tensor) – the tensor to split
# chunks (int) – number of chunks to return（分割的块数）
# dim (int) – dimension along which to split the tensor（沿着哪个轴分块）
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
## FeedForward （GDFN）##
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim * ffn_expansion_factor) #（压缩因子×dim）取整

#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias) # 1×1卷积

#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)

#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x) # 通过1×1卷积之后输出   输入：dim   输出：2×（压缩因子×dim）取整
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 沿1轴（纵向）分为2块
#         x = F.gelu(x1) * x2 # x1的结果通过激活函数之后乘x2
#         x = self.project_out(x) # 1×1卷积
#         return x


    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        # Calculate the hidden features based on the expansion factor
        hidden_features = int(dim * ffn_expansion_factor)

        # 1x1 convolution to project the input to a higher-dimensional space
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 1x1 convolution to project the output back to the original dimension
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)  # Project the input using 1x1 convolution
        x = F.gelu(x)           # Apply GELU activation function
        x = self.project_out(x) # Project back to the original dimension using 1x1 convolution
        return x


#########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
# Transformer里面算自注意力的块（MDTA）
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads  # 头的数量
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 输入：dim 输出：dim*3 1change1卷积
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
#                                     bias=bias)  # 3×3卷积
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 1×1卷积

#     def forward(self, x):
#         b, c, h, w = x.shape  # 输入的维度

#         qkv = self.qkv_dwconv(self.qkv(x))  # 通过1×1卷积之后通过3×3卷积
#         q, k, v = qkv.chunk(3, dim=1)  # 在给定的维度上将张量进行分块 沿1轴（纵向）分3份

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # q
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # k
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # v

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)  # 注意力的结果

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)  # 1×1卷积进行输出维度变换
#         return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, dilation=[1, 2, 3]):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # Number of attention heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 1x1 convolution to produce Q, K, V

        # Create a list of 3x3 convolutional layers with different dilation rates
        self.qkv_dwconv = nn.ModuleList([nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=d, groups=dim * 3,
                                    bias=bias, dilation=d) for d in dilation])

        # Adjusting the input channel size of the project_out layer to match the number of qkv_dwconv layers
        self.project_out = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)  # 1x1 convolution for final output

    def forward(self, x):
        b, c, h, w = x.shape  # Input shape
        qkv = self.qkv(x)  # Pass through 1x1 convolution

        # Pass through 3x3 convolutional layers with different dilation rates
        qkv = [dwconv(qkv) for dwconv in self.qkv_dwconv]
        outs = []
        for qkvi in qkv:
            q, k, v = qkvi.chunk(3, dim=1)  # Split Q, K, V
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)  # Normalize Q
            k = torch.nn.functional.normalize(k, dim=-1)  # Normalize K

            attn = (q @ k.transpose(-2, -1)) * self.temperature  # Compute attention
            attn = attn.softmax(dim=-1)

            out = (attn @ v)  # Compute attention output
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            outs.append(out)  # Append to outputs list

        out_concat = torch.cat(outs, dim=1)  # Concatenate outputs on channel dimension
        out_concat = self.project_out(out_concat)  # Pass through final 1x1 convolution

        return out_concat

##########################################################################
#### Transformer块 #####
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)  # 归一化
        self.attn = Attention(dim, num_heads, bias)  # 计算注意力
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # 层归一化
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # FeedForward

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        print(x.shape)
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
## 重叠的Patch编码 ##
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)  # 3×3卷积

    def forward(self, x):
        x = self.proj(x)

        return x


if __name__=='__main__':
    # model =Attention(dim=64, num_heads=8, bias=True)
    model = TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2.66, bias=False,LayerNorm_type='WithBias')
    x = torch.randn(1,64,128,128)
    y = model(x)
    print(y.shape)