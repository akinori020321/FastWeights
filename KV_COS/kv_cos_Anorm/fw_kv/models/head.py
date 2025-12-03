import torch.nn as nn

class OutputHead(nn.Module):
    def __init__(self, d_h: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_h, num_classes)
    def forward(self, h_T):
        return self.fc(h_T)