import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=0, bias=False),  # 1*8*3*3 = 72
            nn.ReLU(),
            nn.BatchNorm2d(6),                                                                    # 8*2 = 16
            nn.Dropout(dropout_value)
        ) # input: 28x28x1 -> output: 26x26x8, params = 88

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(3, 3), padding=0, bias=False), # 8*32*3*3 = 2,304
            nn.ReLU(),
            nn.BatchNorm2d(32),                                                                    # 32*2 = 64
            nn.Dropout(dropout_value)
        ) # input: 26x26x8 -> output: 24x24x32, params = 2,368

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=6, kernel_size=(1, 1), padding=0, bias=False), # 32*8*1*1 = 256
        ) # input: 24x24x32 -> output: 24x24x8, params = 256
        self.pool1 = nn.MaxPool2d(2, 2) # input: 24x24x8 -> output: 12x12x8, params = 0

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # 8*16*3*3 = 1,152
            nn.ReLU(),            
            nn.BatchNorm2d(16),                                                                    # 16*2 = 32
            nn.Dropout(dropout_value * 1)  # 0.10 dropout
        ) # input: 12x12x8 -> output: 10x10x16, params = 1,184

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # 16*16*3*3 = 2,304
            nn.ReLU(),            
            nn.BatchNorm2d(16),                                                                    # 16*2 = 32
            nn.Dropout(dropout_value * 1.2)  # 0.12 dropout
        ) # input: 10x10x16 -> output: 8x8x16, params = 2,336

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # 16*16*3*3 = 2,304
            nn.ReLU(),            
            nn.BatchNorm2d(16),                                                                    # 16*2 = 32
            nn.Dropout(dropout_value * 1.5)  # 0.15 dropout
        ) # input: 8x8x16 -> output: 8x8x16, params = 2,336
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)                                                           # params = 0
        ) # input: 8x8x16 -> output: 1x1x16

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), # 16*10*1*1 = 160
        ) # input: 1x1x16 -> output: 1x1x10, params = 160

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)       # 28x28 -> 26x26
        x = self.convblock2(x)       # 26x26 -> 24x24
        x = self.convblock3(x)       # 24x24 -> 24x24
        x = self.pool1(x)            # 24x24 -> 12x12
        x = self.convblock4(x)       # 12x12 -> 10x10
        x = self.convblock5(x)       # 10x10 -> 8x8
        x = self.convblock7(x)       # 8x8 -> 8x8
        x = self.gap(x)              # 8x8 -> 1x1
        x = self.convblock8(x)       # 1x1 -> 1x1

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Total Parameter Count:
# convblock1:      88 (72 + 16)        # Conv + BN
# convblock2:   2,368 (2,304 + 64)     # Conv + BN
# convblock3:     256 (256 + 0)        # Only Conv
# pool1:            0                   # No params
# convblock4:   1,184 (1,152 + 32)     # Conv + BN
# convblock5:   2,336 (2,304 + 32)     # Conv + BN
# convblock7:   2,336 (2,304 + 32)     # Conv + BN
# gap:              0                   # No params
# convblock8:     160 (160 + 0)        # Only Conv
# Total Parameters: 8,728