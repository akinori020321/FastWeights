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

# class OutputHead(nn.Module):
#     """
#     方向復元タスク専用ヘッド：
#     d_h → hidden → d_g の 1層MLP
#     """
#     def __init__(self, d_h: int, d_g: int, hidden_dim: int = 128):
#         super().__init__()
#         self.fc1 = nn.Linear(d_h, hidden_dim)
#         self.act = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, d_g)

#     def forward(self, h_T):
#         # h_T: (B, d_h)
#         x = self.fc1(h_T)
#         x = self.act(x)
#         x = self.fc2(x)
#         return x  # (B, d_g)
