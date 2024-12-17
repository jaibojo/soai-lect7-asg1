import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # Block 1: Input Block (1 conv layer)
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),    # params: (1*8*3*3) + 8 = 80
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU()
        )   # Output: 28x28x8
        
        # Block 2: Block1 (2 conv layers)
        self.block1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU()
        )   # Output: 28x28x8
        
        # Block 3: Transition Block1
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2),                              # params: 0
            nn.Conv2d(8, 8, kernel_size=1),               # params: (8*8*1*1) + 8 = 72
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU()
        )   # Output: 14x14x8
        
        # Block 4: Block2 (2 conv layers)
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # params: (8*16*3*3) + 16 = 1,168
            nn.BatchNorm2d(16),                           # params: 16*2 = 32
            nn.ReLU()
        )   # Output: 14x14x16
        
        # Block 5: Transition Block2
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2),                              # params: 0
            nn.Conv2d(16, 8, kernel_size=1),              # params: (16*8*1*1) + 8 = 136
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU()
        )   # Output: 7x7x8
        
        # Block 6: Block3 (2 conv layers)
        self.block3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # params: (8*16*3*3) + 16 = 1,168
            nn.BatchNorm2d(16),                           # params: 16*2 = 32
            nn.ReLU()
        )   # Output: 7x7x16
        
        # Block 7: Output Block
        self.output_block = nn.Sequential(
            nn.MaxPool2d(2),                              # Output: 3x3x16
            nn.Flatten(),                                 # Output: 144
            nn.Linear(16 * 3 * 3, 10)                    # params: (16*3*3*10) + 10 = 1,450
        )
        
    def forward(self, x):
        x = self.input_block(x)      # Input Block
        x = self.block1(x)           # Block1
        x = self.transition1(x)      # Transition1
        x = self.block2(x)           # Block2
        x = self.transition2(x)      # Transition2
        x = self.block3(x)           # Block3
        x = self.output_block(x)     # Output Block
        return x 

# Total Model Parameters: 8,338
# Block-wise Parameter Distribution:
# - Input Block:      96 parameters    (1.2%)
# - Block1:        1,200 parameters   (14.4%)
# - Transition1:      88 parameters    (1.1%)
# - Block2:        1,800 parameters   (21.6%)
# - Transition2:     152 parameters    (1.8%)
# - Block3:        3,552 parameters   (42.6%)
# - Output Block:  1,450 parameters   (17.4%)
#
# Architecture Summary:
# Input: 28x28x1 
# After Input Block:   28x28x8
# After Block1:        28x28x8
# After Transition1:   14x14x8
# After Block2:        14x14x16
# After Transition2:   7x7x8
# After Block3:        7x7x16
# After Output Block:  10 (classes)
#
# BatchNorm Details:
# - Each BatchNorm2d layer adds 2*channels parameters
# - Total BatchNorm params: 192 (2.3% of total)
# - Added after every Conv2d except final layer
# - Helps with:
#   1. Faster training convergence
#   2. Reduces internal covariate shift
#   3. Acts as regularization