import torch
from torch import nn, einsum
from torchvision.ops import generalized_box_iou
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..utils.utils import box_xywh_to_xyxy


def loss_contrastive(l_text_latents, l_image_latents, g_text_latents, g_image_latents, global_rank, temperature):
    b = l_text_latents.shape[0]
    sim_text = einsum('i d, j d -> i j', l_text_latents, g_image_latents) * temperature  # [local_bs, global_bs]
    sim_image = einsum('i d, j d -> i j', l_image_latents, g_text_latents) * temperature  # [local_bs, global_bs]
    labels = torch.arange(global_rank * b, (global_rank + 1) * b).to(l_text_latents.device)  # local_bs
    loss_i2t = F.cross_entropy(sim_image, labels)
    loss_t2i = F.cross_entropy(sim_text, labels)
    loss = (loss_t2i + loss_i2t) / 2.0
    return loss


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class DetectionCriterion(nn.Module):

    def __init__(self, losses):
        super().__init__()
        self.losses = losses

    # возвращает индексы предсказанных боксов, у которых IoU с ground truth наибольшее
    @torch.no_grad()
    def _get_idx(self, outputs, targets):
        pred_boxes = outputs["pred_boxes"]
        pred_classes = outputs["pred_classes"]

        giou_parts = [-generalized_box_iou(box_xywh_to_xyxy(x1), box_xywh_to_xyxy(x2)) for x1, x2 in
                      zip(pred_boxes, targets)]
        class_parts = [-pred_classes[i].unsqueeze(-1).repeat(1, targets[i].shape[0]) for i in range(len(targets))]
        bbox_parts = [torch.cdist(x1, x2, p=1) for x1, x2 in zip(pred_boxes, targets)]

        costs = [
            class_part + giou_part + bbox_part
            for class_part, giou_part, bbox_part in zip(class_parts, giou_parts, bbox_parts)
        ]
        boxes_idx = [linear_sum_assignment(cost.detach().cpu())[0] for cost in costs]

        return boxes_idx

    #  для боксов, полученных при помощи _get_idx предсказываем 1, для остальных 0
    def loss_classification(self, outputs, targets, boxes_idx, num_boxes):
        pred_probs = outputs["pred_classes"]
        target_labels = [torch.zeros_like(pred_prob) for pred_prob in pred_probs]
        for i, idx in enumerate(boxes_idx):
            target_labels[i][idx] = 1.

        pred_probs = pred_probs.reshape(-1)
        target_labels = torch.cat(target_labels)
        loss = F.binary_cross_entropy(pred_probs, target_labels)

        return {"loss_classification": loss}

    # L1 и GIoU между боксами, полученными при помощи _get_idx и ground truth
    def loss_boxes(self, outputs, targets, boxes_idx, num_boxes):
        src_boxes = torch.cat([t[idx] for t, idx in zip(outputs["pred_boxes"], boxes_idx)], dim=0)
        target_boxes = torch.cat([t for t in targets], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1. - torch.diag(
            generalized_box_iou(
                box_xywh_to_xyxy(src_boxes), box_xywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def get_loss(self, loss, outputs, targets, boxes_idx, num_boxes, **kwargs):
        loss_map = {
            "classification": self.loss_classification,
            "boxes": self.loss_boxes
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"

        return loss_map[loss](outputs, targets, boxes_idx, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        boxes_idx = self._get_idx(outputs["pred_boxes"], targets)
        num_boxes = float(sum([boxes.shape[0] for boxes in targets]))

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, boxes_idx, num_boxes))

        return losses
