"""
Docstring for temporalgan.gen_s2_lc_v1_1
Version 1.1: Inheritance from Version 1.0 
- Add CBAM at each skip connections
"""

import torch
import torch.nn as nn
from submodules.cbam import ChannelAttention, CBAM
from submodules.gen_cnn_block import Block


class Generator(nn.Module):
    def __init__(self, s2_in_channels = 3, lc_in_channels = 3, out_channels = 1, features = 64):
        super().__init__()
        S2_IN_OUT_KSP = (3, 1, 1)
        LC_IN_OUT_KSP = (5, 1, 2)
        OUTPUT_KSP = (3, 1, 1)

        self.s2_initial_down = nn.Sequential(
            nn.Conv2d(in_channels=s2_in_channels, out_channels=features, kernel_size=S2_IN_OUT_KSP[0], stride=S2_IN_OUT_KSP[1], padding=S2_IN_OUT_KSP[2]),
            nn.LeakyReLU(0.2)
        )

    
        self.lc_initial_down = nn.Sequential(
            nn.Conv2d(in_channels=lc_in_channels, out_channels=features, kernel_size=LC_IN_OUT_KSP[0], stride=LC_IN_OUT_KSP[1], padding=LC_IN_OUT_KSP[2]),
            nn.LeakyReLU(0.2)
        )

        # Downsample blocks of Sen2
        self.cbam_init_s2 = CBAM(channel= features)
        self.down1_s2 = Block(features, features * 2, down=True, act='leaky', use_dropout=False)
        self.cbam1_s2 = CBAM(channel=features * 2)
        self.down2_s2 = Block(features*2, features*4, down=True, act='leaky', use_dropout=False)
        self.cbam2_s2 = CBAM(channel= features * 4)
        self.down3_s2 = Block(features*4, features*8, down=True, act = 'leaky', use_dropout=False)
        self.cbam3_s2 = CBAM(channel=features*8)
        self.down4_s2 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False)
        self.cbam4_s2 = CBAM(channel=features * 8)
        self.down5_s2 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False)
        self.cbam5_s2 = CBAM(channel=features * 8)
        self.down6_s2 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False)

        # Downsample blocks of LC
        self.cbam_init_lc = CBAM(channel=features)
        self.down1_lc = Block(features, features*2, down=True, act='leaky', use_dropout=False)
        self.cbam1_lc = CBAM(channel=features*2)
        self.down2_lc = Block(features*2, features*4, down = True, act = 'leaky', use_dropout=False)
        self.cbam2_lc = CBAM(channel=features * 4)
        self.down3_lc = Block(features*4, features*8, down=True, act='leaky', use_dropout=False)
        self.cbam3_lc = CBAM(channel= features * 8)
        self.down4_lc = Block(features * 8, features *8, down=True, act='leaky', use_dropout=False)
        self.cbam4_lc = CBAM(channel = features * 8)
        self.down5_lc = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
        self.cbam5_lc = CBAM(channel = features * 8)
        self.down6_lc = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)

        # Channel attention
        self.ca = ChannelAttention(n_channels = features*8*2)

        # Bottleneck layer (downsampling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8*2, features*8, 4, 2, 1),
            nn.ReLU()
        )

        # Upsample blocks
        self.up1 = Block(features*8, features * 8, down=False, act='relu', use_dropout=True)
        self.up2 = Block(features*8 * 3, features * 8, down=False, act='relu', use_dropout=True)
        self.up3 = Block(features*8*3, features * 8, down=False, act='relu', use_dropout=True)
        self.up4 = Block(features*8*3, features*8, down=False, act='relu', use_dropout=False)
        self.up5 = Block(features*8*3, features*4, down=False, act='relu', use_dropout=False)
        self.up6 = Block(features*4*3, features*2, down=False, act='relu', use_dropout=False)
        self.up7 = Block(features*2*3, features, down=False, act='relu', use_dropout=False)

        self.final_up = nn.Sequential(
            nn.Conv2d(features*3, out_channels, OUTPUT_KSP[0], OUTPUT_KSP[1], OUTPUT_KSP[2]),
            nn.Tanh()
        )

    
    def forward(self, s2: torch.Tensor, lc:torch.Tensor) -> torch.Tensor:
        # First we do the encoding part for the S2
        d1_s2 = self.s2_initial_down(s2)
        d2_s2 = self.down1_s2(d1_s2)
        d3_s2 = self.down2_s2(d2_s2)
        d4_s2 = self.down3_s2(d3_s2)
        d5_s2 = self.down4_s2(d4_s2)
        d6_s2 = self.down5_s2(d5_s2)
        d7_s2 = self.down6_s2(d6_s2)

        # The encoding part for the LC
        d1_lc = self.lc_initial_down(lc)
        d2_lc = self.down1_lc(d1_lc)
        d3_lc = self.down2_lc(d2_lc)
        d4_lc = self.down3_lc(d3_lc)
        d5_lc = self.down4_lc(d4_lc)
        d6_lc = self.down5_lc(d5_lc)
        d7_lc = self.down6_lc(d6_lc)

        # Fuse the two streams
        d7 = torch.cat([d7_s2, d7_lc], dim = 1)
        
        # channel attention
        d7 = self.ca(d7)
        
        # bottleneck
        bottleneck = self.bottleneck(d7)
        
        # decoding part + cbam at skip connection
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, self.cbam5_s2(d6_s2), self.cbam5_lc(d6_lc)], 1))
        up4 = self.up4(torch.cat([up3, self.cbam4_s2(d5_s2), self.cbam4_lc(d5_lc)], 1))
        up5 = self.up5(torch.cat([up4, self.cbam3_s2(d4_s2), self.cbam3_lc(d4_lc)], 1))
        up6 = self.up6(torch.cat([up5, self.cbam2_s2(d3_s2), self.cbam2_lc(d3_lc)], 1))
        up7 = self.up7(torch.cat([up6, self.cbam1_s2(d2_s2), self.cbam1_lc(d2_lc)], 1))

        # final upsampling
        return self.final_up(torch.cat([up7, self.cbam_init_s2(d1_s2), self.cbam_init_lc(d1_lc)], 1))
    
if __name__ == "__main__":
    s2 = torch.rand(4, 3, 256, 256)
    lc = torch.rand(4, 3, 256, 256)

    netG = Generator(3, 3, 1, 64)

    s1_out = netG(s2, lc)
    print(s1_out.shape)