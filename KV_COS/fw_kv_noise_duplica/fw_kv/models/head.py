import torch.nn as nn
import torch
import torch.nn.functional as F


class OutputHead(nn.Module):
    """
    方向復元タスク専用ヘッド：
    d_h → d_g へ線形変換
    """
    def __init__(self, d_h: int, d_g: int):
        super().__init__()
        self.fc = nn.Linear(d_h, d_g)

    def forward(self, h_T):
        # h_T: (B, d_h)
        return self.fc(h_T)   # (B, d_g)
