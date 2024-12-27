import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=0, bias=False),  # 1*6*3*3 = 54
            nn.ReLU(),
            nn.BatchNorm2d(6),                                                                    # 6*2 = 12
            nn.Dropout(dropout_value)
        ) # input: 28x28x1 -> output: 26x26x6, RF: 3x3, params = 66

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(3, 3), padding=0, bias=False), # 6*32*3*3 = 1,728
            nn.ReLU(),
            nn.BatchNorm2d(32),                                                                    # 32*2 = 64
            nn.Dropout(dropout_value)
        ) # input: 26x26x6 -> output: 24x24x32, RF: 5x5, params = 1,792

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=6, kernel_size=(1, 1), padding=0, bias=False), # 32*6*1*1 = 192
        ) # input: 24x24x32 -> output: 24x24x6, RF: 5x5, params = 192
        self.pool1 = nn.MaxPool2d(2, 2) # input: 24x24x6 -> output: 12x12x6, RF: 6x6, params = 0

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # 6*16*3*3 = 864
            nn.ReLU(),            
            nn.BatchNorm2d(16),                                                                    # 16*2 = 32
            nn.Dropout(dropout_value)
        ) # input: 12x12x6 -> output: 10x10x16, RF: 10x10, params = 896

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # 16*16*3*3 = 2,304
            nn.ReLU(),            
            nn.BatchNorm2d(16),                                                                    # 16*2 = 32
            nn.Dropout(dropout_value * 1.2)
        ) # input: 10x10x16 -> output: 8x8x16, RF: 14x14, params = 2,336

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # 16*16*3*3 = 2,304
            nn.ReLU(),            
            nn.BatchNorm2d(16),                                                                    # 16*2 = 32
            nn.Dropout(dropout_value * 1.5)
        ) # input: 8x8x16 -> output: 8x8x16, RF: 18x18, params = 2,336

        self.gap = nn.AdaptiveAvgPool2d(1)  # input: 8x8x16 -> output: 1x1x16, RF: 26x26, params = 0
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), # 16*10*1*1 = 160
        ) # input: 1x1x16 -> output: 1x1x10, RF: 26x26, params = 160

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)       # 28x28 -> 26x26, RF: 3
        x = self.convblock2(x)       # 26x26 -> 24x24, RF: 5
        x = self.convblock3(x)       # 24x24 -> 24x24, RF: 5
        x = self.pool1(x)            # 24x24 -> 12x12, RF: 6
        x = self.convblock4(x)       # 12x12 -> 10x10, RF: 10
        x = self.convblock5(x)       # 10x10 -> 8x8, RF: 14
        x = self.convblock7(x)       # 8x8 -> 8x8, RF: 18
        x = self.gap(x)              # 8x8 -> 1x1, RF: 26
        x = self.convblock8(x)       # 1x1 -> 1x1, RF: 26

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Layer-wise Analysis:
# Layer     RF    Nin  Nout   Comment
# Input     1x1   28   28     Input Image
# Conv1     3x3   28   26     3x3 kernel
# Conv2     5x5   26   24     +2 from 3x3 kernel
# Conv3     5x5   24   24     1x1 kernel doesn't change RF
# Pool1     6x6   24   12     Maxpool doubles jump
# Conv4    10x10  12   10     +4 from 3x3 kernel (doubled jump)
# Conv5    14x14  10   8      +4 from 3x3 kernel (doubled jump)
# Conv7    18x18   8   8      +4 from 3x3 kernel (doubled jump)
# GAP      26x26   8   1      Covers entire remaining feature map
# Conv8    26x26   1   1      1x1 kernel doesn't change RF

# Total Parameter Count:
# convblock1:      66 (54 + 12)        # Conv + BN
# convblock2:   1,792 (1,728 + 64)     # Conv + BN
# convblock3:     192 (192 + 0)        # Only Conv
# pool1:            0                   # No params
# convblock4:     896 (864 + 32)       # Conv + BN
# convblock5:   2,336 (2,304 + 32)     # Conv + BN
# convblock7:   2,336 (2,304 + 32)     # Conv + BN
# gap:              0                   # No params
# convblock8:     160 (160 + 0)        # Only Conv
# Total Parameters: 7,778