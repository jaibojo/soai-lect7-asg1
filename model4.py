import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self, dropout_rate=0.10):
        super(MNISTClassifier, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),  # 1*8*3*3 = 72
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        ) # output_size = 28x28x8, RF = 3, params = 88

        # Block1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),  # 8*32*3*3 = 2,304
            nn.BatchNorm2d(32),                                                                    # 32*2 = 64
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        ) # output_size = 28x28x32, RF = 5, params = 2,368

        # Transition Block1
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), bias=False),           # 32*8*1*1 = 256
            nn.MaxPool2d(2, 2),                                                                   # params = 0
        ) # output_size = 14x14x8, RF = 6, params = 256

        # Block2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),  # 8*8*3*3 = 576
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        ) # output_size = 14x14x8, RF = 10, params = 592


        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # 8*32*3*3 = 2,304
            nn.BatchNorm2d(32),                                                                    # 32*2 = 64
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        ) # output_size = 14x14x32, RF = 18, params = 4,672
           
        self.gap.AdaptiveAvgPool2d(1)                                                               # params = 0

        # Output Block
        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), bias=False),           # 32*10*1*1 = 320
        ) # output_size = 1x1x10, RF = Global, params = 320

    def forward(self, x):
        x = self.convblock1(x)       # Input Block
        x = self.convblock2(x)       # First Conv
        x = self.transition1(x)      # Transition Block1
        x = self.convblock4(x)       # Third Conv
        x = self.convblock6(x)       # Sixth Conv
        x = self.output_block(x)     # Output Block
        x = x.view(-1, 10)           # Reshape to (batch_size, 10)
        return F.log_softmax(x, dim=-1)

# Parameter Count:
# convblock1:      88 (72 + 16)       # 1*8*3*3 = 72 conv + 8*2 BN
# convblock2:   2,368 (2,304 + 64)    # 8*32*3*3 = 2,304 conv + 32*2 BN
# transition1:    256 (256 + 0)       # 32*8*1*1 = 256 conv (no BN)
# convblock4:     592 (576 + 16)      # 8*8*3*3 = 576 conv + 8*2 BN
# convblock6:   4,672 (4,608 + 64)    # 8*32*3*3 = 4,608 conv + 32*2 BN
# output_block:   320 (320 + 0)       # 32*10*1*1 = 320 conv (no BN)
# Total Parameters: 8,296

# Architecture Features:
# 1. All convolutions are 3x3 except in transition and output (1x1)
# 2. BatchNorm after every convolution (except transition and output)
# 3. Dropout (10%) after every ReLU except output block
# 4. Global Average Pooling in output block
# 5. No bias in convolutions to reduce parameters
# 6. Log softmax for better numerical stability

# Channel progression:
# Input:        1
# Block1:       8 -> 32
# Transition1:  32 -> 8
# Block2:       8 -> 8
# Block3:       8 -> 32
# Output:       32 -> 10

# Receptive Field Calculation:
# Layer               RF      Jump    RF Calc
# Input              1       1       -
# convblock1         3       1       1 + (3-1)
# convblock2         5       1       3 + (3-1)
# MaxPool2d          6       2       5 + 1
# convblock4         10      2       6 + 2*(3-1)
# convblock6         18      2       10 + 2*(3-1)
# GAP                Global   -       Global