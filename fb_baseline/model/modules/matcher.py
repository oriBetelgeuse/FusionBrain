import torch
from torch import nn
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment

from ..utils.utils import box_xywh_to_xyxy


class HungarianMatcher(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, targets):
        pass
