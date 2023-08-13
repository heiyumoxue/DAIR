import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.archs import LXS_former_block as MF



@ARCH_REGISTRY.register()  #这里是注册  单独调试代码的时候可以注释这行  但是整体运行的时候记得去掉注释
class LXS_former(nn.Module):
    def __init__(self, in_nc=3, nf=100, num_modules=4, out_nc=3, upscale=2):
        super(LXS_former, self).__init__()

        self.fea_conv = MF.conv_layer(in_nc, nf, kernel_size=3)  # mobile_former

        self.MF1 = MF.TransformerBlock(dim=nf, num_heads=2, ffn_expansion_factor=2.66, bias=False,LayerNorm_type='WithBias')
        self.MF2 = MF.TransformerBlock(dim=nf, num_heads=2, ffn_expansion_factor=2.66, bias=False,LayerNorm_type='WithBias')
        self.MF3 = MF.TransformerBlock(dim=nf, num_heads=2, ffn_expansion_factor=2.66, bias=False,LayerNorm_type='WithBias')
        self.MF4 = MF.TransformerBlock(dim=nf, num_heads=2, ffn_expansion_factor=2.66, bias=False,LayerNorm_type='WithBias')

        self.c = MF.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = MF.conv_layer(nf, nf, kernel_size=3)
        upsample_block = MF.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_MF1 = self.MF1(out_fea)
        out_MF2 = self.MF2(out_MF1)
        out_MF3 = self.MF3(out_MF2)
        out_MF4 = self.MF4(out_MF3)


        # print(out_MF.shape)
        out_B = self.c(torch.cat([out_MF1, out_MF2, out_MF3, out_MF4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


if __name__ == '__main__':
    x = torch.randn((16,3,64,64))
    model = LXS_former(in_nc=3, out_nc=3,nf=50,upscale=2)
    print(model)
    out = model(x)
    print(out.shape)
