import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):  # Removed dropout_rate parameter
        super(MNISTClassifier, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),  # 1*8*3*3 = 72
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU()
        ) # output_size = 28x28x8, RF = 3

        # Block1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),  # 8*8*3*3 = 576
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU()
        ) # output_size = 28x28x8, RF = 5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # 8*16*3*3 = 1,152
            nn.BatchNorm2d(16),                                                                   # 16*2 = 32
            nn.ReLU()
        ) # output_size = 28x28x16, RF = 7

        # Transition Block1
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                                                   # RF increases by 1
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), bias=False),           # 16*8*1*1 = 128
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU()
        ) # output_size = 14x14x8, RF = 8

        # Block2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),  # 8*8*3*3 = 576
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU()
        ) # output_size = 14x14x8, RF = 12

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # 8*16*3*3 = 1,152
            nn.BatchNorm2d(16),                                                                   # 16*2 = 32
            nn.ReLU()
        ) # output_size = 14x14x16, RF = 16

        # Transition Block2
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                                                   # RF increases by 1
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), bias=False),           # 16*8*1*1 = 128
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU()
        ) # output_size = 7x7x8, RF = 17

        # Block3
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),  # 8*8*3*3 = 576
            nn.BatchNorm2d(8),                                                                    # 8*2 = 16
            nn.ReLU()
        ) # output_size = 7x7x8, RF = 25

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # 8*16*3*3 = 1,152
            nn.BatchNorm2d(16),                                                                   # 16*2 = 32
            nn.ReLU()
        ) # output_size = 7x7x16, RF = 33

        # Output Block
        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), bias=False),           # 16*10*1*1 = 160
            nn.AdaptiveAvgPool2d(1)                                                              # Global RF after GAP
        ) # output_size = 1x1x10, RF = Global

    def forward(self, x):
        x = self.convblock1(x)       # Input Block
        x = self.convblock2(x)       # First Conv
        x = self.convblock3(x)       # Second Conv
        x = self.transition1(x)      # Transition Block1
        x = self.convblock4(x)       # Third Conv
        x = self.convblock5(x)       # Fourth Conv
        x = self.transition2(x)      # Transition Block2
        x = self.convblock6(x)       # Fifth Conv
        x = self.convblock7(x)       # Sixth Conv
        x = self.output_block(x)     # Output Block
        x = x.view(-1, 10)           # Reshape to (batch_size, 10)
        return F.log_softmax(x, dim=-1)

# Parameter Count (unchanged):
# convblock1:      88 (72 + 16)
# convblock2:     592 (576 + 16)
# convblock3:   1,184 (1,152 + 32)
# transition1:    144 (128 + 16)
# convblock4:     592 (576 + 16)
# convblock5:   1,184 (1,152 + 32)
# transition2:    144 (128 + 16)
# convblock6:     592 (576 + 16)
# convblock7:   1,184 (1,152 + 32)
# output_block:   180 (160 + 20)
# Total Parameters: 5,884

# Architecture Features:
# 1. All convolutions are 3x3 except in transition and output (1x1)
# 2. BatchNorm after every convolution
# 3. No Dropout (removed)
# 4. Global Average Pooling in output block
# 5. No bias in convolutions to reduce parameters
# 6. Log softmax for better numerical stability
#
# Receptive Field Calculation:
# Layer               RF      Jump    RF Calc
# Input              1       1       -
# convblock1         3       1       1 + (3-1)
# convblock2         5       1       3 + (3-1)
# convblock3         7       1       5 + (3-1)
# MaxPool2d          8       2       7 + 1
# convblock4         12      2       8 + 2*(3-1)
# convblock5         16      2       12 + 2*(3-1)
# MaxPool2d          17      4       16 + 1
# convblock6         25      4       17 + 4*(3-1)
# convblock7         33      4       25 + 4*(3-1)
# GAP                Global   -       Global
#
# Note: 
# - RF increases by (kernel_size-1) for each conv layer
# - After MaxPool, jump multiplies by 2
# - MaxPool adds 1 to RF but multiplies the jump 