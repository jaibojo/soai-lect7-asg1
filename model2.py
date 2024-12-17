import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, dropout_rate=0.10):
        super(MNISTClassifier, self).__init__()
        
        # Block 1: Input Block (1 conv layer)
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),    # params: (1*8*3*3) + 8 = 80
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)                    # No params
        )   # Output: 28x28x8
        
        # Block 2: Block1 (2 conv layers)
        self.block1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),                   # No params
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)                    # No params
        )   # Output: 28x28x8
        
        # Block 3: Transition Block1
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2),                              # params: 0
            nn.Conv2d(8, 8, kernel_size=1),               # params: (8*8*1*1) + 8 = 72
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)                    # No params
        )   # Output: 14x14x8
        
        # Block 4: Block2 (2 conv layers)
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),                   # No params
            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # params: (8*16*3*3) + 16 = 1,168
            nn.BatchNorm2d(16),                           # params: 16*2 = 32
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)                    # No params
        )   # Output: 14x14x16
        
        # Block 5: Transition Block2
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2),                              # params: 0
            nn.Conv2d(16, 8, kernel_size=1),              # params: (16*8*1*1) + 8 = 136
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)                    # No params
        )   # Output: 7x7x8
        
        # Block 6: Block3 (2 conv layers)
        self.block3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),    # params: (8*8*3*3) + 8 = 584
            nn.BatchNorm2d(8),                            # params: 8*2 = 16
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),                   # No params
            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # params: (8*16*3*3) + 16 = 1,168
            nn.BatchNorm2d(16),                           # params: 16*2 = 32
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)                    # No params
        )   # Output: 7x7x16
        
        # Block 7: Output Block (no dropout here)
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

# Total Model Parameters: 6,586
# Block-wise Parameter Distribution:
# - Input Block:      96 parameters    (1.5%)
# - Block1:        1,200 parameters   (18.2%)
# - Transition1:      88 parameters    (1.3%)
# - Block2:        1,800 parameters   (27.3%)
# - Transition2:     152 parameters    (2.3%)
# - Block3:        1,800 parameters   (27.3%)
# - Output Block:  1,450 parameters   (22.0%)
#
# Architecture Summary:
# Input: 28x28x1 
# After Input Block:   28x28x8   + Dropout(0.10)
# After Block1:        28x28x8   + Dropout(0.10)
# After Transition1:   14x14x8   + Dropout(0.10)
# After Block2:        14x14x16  + Dropout(0.10)
# After Transition2:   7x7x8     + Dropout(0.10)
# After Block3:        7x7x16    + Dropout(0.10)
# After Output Block:  10 (classes)
#
# Layer Details:
# - BatchNorm after each Conv2d (except output)
# - Dropout(0.10) after each ReLU (except output)
# - Total trainable params unchanged (dropout has no parameters)
# - Dropout helps prevent overfitting by:
#   1. Randomly zeroing entire feature maps (Dropout2d)
#   2. Creating ensemble-like behavior
#   3. Reducing co-adaptation of neurons