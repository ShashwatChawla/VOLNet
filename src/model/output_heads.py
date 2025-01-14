import torch

import torch.nn as nn
import torch.nn.functional as F

class RotationHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RotationHead, self).__init__()
        # 3 Layer FC w/t ReLU activation
        self.rotation_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size // 4), 
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
        
    def forward(self, x):
        return self.rotation_head(x)

class TranslationHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TranslationHead, self).__init__()
        # 3 Layer FC w/t ReLU activation
        self.translation_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size // 4), 
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, x):
        return self.translation_head(x)