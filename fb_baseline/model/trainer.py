from collections import OrderedDict
from abc import abstractmethod

import torch
from torch import nn
import pytorch_lightning as pl

from modules.loss import DetectionCriterion, loss_contrastive


class BaseTrainer(pl.LightningModule):

    def __init__(self, model, config, ctc_labeling):
        super().__init__()
        self.model = model
        self.model.freeze_gpt(**config['freeze'])
        if config['weights_path'] is not None:
            state_dict = torch.load(config['weights_path'])['state_dict']
            state_dict = OrderedDict({key[6:]: value for key, value in state_dict.items()})
            self.model.load_state_dict(state_dict)
        self.config = config

        self.handwritten_criterion = torch.nn.CTCLoss(zero_infinity=True)
        self.ctc_labeling = ctc_labeling
        self.c2c_criterion = nn.CrossEntropyLoss()
        self.detection_criterion = DetectionCriterion(config['detection_losses'])
        self.detection_losses_weights = config['detection_losses_weights']
        self.vqa_criterion = nn.CrossEntropyLoss(ignore_index=config['ignore_index'])

    def forward(self, task_id, **kwargs):
        out = self.model(task_id, **kwargs)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['max_lr'],
            total_steps=self.config['total_steps'],
            pct_start=self.config['pct_start'],
        )
        sched = {
            'scheduler': scheduler,
            'interval': 'step',
        }
        return [optimizer], [sched]

    def htr_step(self, htr_images, encoded, encoded_length):
        bs = htr_images.shape[0]
        images = htr_images.type(torch.float32)
        encoded_length = encoded_length.type(torch.int32)
        encoded = encoded.type(torch.int32)
        handwritten_outputs = self.model('handwritten', images=images)
        preds_size = torch.IntTensor([handwritten_outputs.size(1)] * bs)
        preds = handwritten_outputs.log_softmax(2).permute(1, 0, 2)
        handwritten_loss = self.handwritten_criterion(preds, encoded, preds_size, encoded_length)
        return handwritten_loss

    def trans_step(self, code_input_ids, code_input_labels):
        code_input_ids = code_input_ids.type(torch.long)
        code_input_labels = code_input_labels.type(torch.long)
        loss_mask = torch.tensor(code_input_labels.clone().detach() == 2, dtype=torch.uint8)
        lm_logits = self.model('trans', input_ids=code_input_ids, input_labels=code_input_labels)
        c_labels = code_input_ids
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = c_labels[..., 1:].contiguous()

        flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
        c2c_loss = self.c2c_criterion(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
        return c2c_loss

    @abstractmethod
    def vqa_step(self):
        return

    @abstractmethod
    def detection_step(self):
        return

    @abstractmethod
    def model_step(self, batch, stage):
        return

    def training_step(self, train_batch, batch_idx):
        return self.model_step(train_batch, 'train')

    def validation_step(self, valid_batch, batch_idx):
        self.model_step(valid_batch, 'valid')


class CrossAttentionTrainer(BaseTrainer):

    def __init__(self, model, config, ctc_labeling):
        super().__init__(model, config, ctc_labeling)
        self.temperature = config['temperature']

    def vqa_step(self, vqa_images, vqa_input_ids, labels, targets):
        images = vqa_images.type(torch.float32)
        input_ids = vqa_input_ids.type(torch.long)
        labels = labels.type(torch.float32)
        loss_mask = torch.tensor(labels.clone().detach() == 2, dtype=torch.uint8)
        vqa_outputs = self.model('vqa', images=images, tokens=input_ids, labels=labels)
        lm_logits = vqa_outputs['pred_logits']
        labels = input_ids
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        local_vqa_proj_tokens, local_vqa_proj_queries = vqa_outputs['proj_tokens'], vqa_outputs['proj_queries']
        flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
        vqa_loss = self.vqa_criterion(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])

        vqa_proj = {
            'local_tokens': local_vqa_proj_tokens,
            'local_queries': local_vqa_proj_queries,
        }
        return vqa_loss, vqa_proj

    def detection_step(self, detection_images, detection_input_ids, detection_attention_masks, boxes):
        images = detection_images.to(self.device, dtype=torch.float32)
        input_ids = detection_input_ids.to(self.device, dtype=torch.long)
        attention_masks = detection_attention_masks.to(self.device, dtype=torch.long)
        boxes = [boxes_per_label.to(self.device, dtype=torch.float) for boxes_per_label in boxes]
        detection_outputs = self.model('detection', images=images, tokens=input_ids, attention_masks=attention_masks)
        local_detection_proj_tokens, local_detection_proj_queries = detection_outputs['proj_tokens'], detection_outputs[
            'proj_queries']

        local_bath_size = detection_outputs['pred_logits'].shape[0]
        detection_loss = self.detection_criterion(
            detection_outputs,
            boxes[self.global_rank * local_bath_size: (self.global_rank + 1) * local_bath_size]
        )
        detection_proj = {
            'local_tokens': local_detection_proj_tokens,
            'local_queries': local_detection_proj_queries,
        }
        return detection_loss, detection_proj

    def model_step(self, batch, stage):
        (htr_images, encoded, encoded_length, gt_texts), (code_input_ids, code_input_labels, code_targets), (
        vqa_images, vqa_input_ids, labels, targets), (
        detection_names, detection_images, detection_input_ids, detection_attention_masks, boxes, size) = batch
        losses = []
        metrics = {}
        local_proj_tokens = []
        local_proj_queries = []
        if len(htr_images) > 0:
            htr_loss = self.htr_step(htr_images, encoded, encoded_length)
            metrics['h_loss'] = htr_loss.detach().cpu().item()
            self.log(f'{stage}_handwritten_loss', metrics['h_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(htr_loss)

        if len(code_input_ids) > 0:
            c2c_loss = self.trans_step(code_input_ids, code_input_labels)
            metrics['c2c_loss'] = c2c_loss.detach().cpu().item()
            self.log(f'{stage}_c2c_loss', metrics['c2c_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(c2c_loss)

        if len(labels) > 0:
            vqa_loss, vqa_proj = self.vqa_step(vqa_images, vqa_input_ids, labels, targets)
            metrics['vqa_loss'] = vqa_loss.detach().cpu().item()
            self.log(f'{stage}_vqa_loss', metrics['vqa_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(vqa_loss)

            local_proj_tokens.append(vqa_proj['local_tokens'])
            local_proj_queries.append(vqa_proj['local_queries'])

        if len(boxes) > 0:
            detection_loss, detection_proj = self.detection_step(
                detection_images, detection_input_ids, detection_attention_masks, boxes
            )
            sum_detection_losses = sum([
                loss * weight for loss, weight in zip(detection_loss.values(), self.detection_losses_weights)
            ])
            self.log(f'{stage}_detection_loss', sum_detection_losses, on_epoch=True, prog_bar=True, logger=True)
            for key in detection_loss:
                metrics[key] = detection_loss[key].detach().cpu().item()
                self.log(f'{stage}_{key}', metrics[key], on_epoch=True, prog_bar=True, logger=True)
            losses.append(sum_detection_losses)

            local_proj_tokens.append(detection_proj['local_tokens'])
            local_proj_queries.append(detection_proj['local_queries'])

        if len(boxes) > 0 or len(labels) > 0:
            local_proj_tokens = torch.cat(local_proj_tokens)
            local_proj_queries = torch.cat(local_proj_queries)
            global_proj_tokens, global_proj_queries = self._all_gather_proj(local_proj_tokens, local_proj_queries)

            contrastive_loss = loss_contrastive(
                local_proj_tokens,
                local_proj_queries,
                global_proj_tokens,
                global_proj_queries,
                self.global_rank,
                self.temperature
            )
            metrics['contrastive_loss'] = contrastive_loss.detach().cpu().item()
            self.log(f'{stage}_contrastive_loss', metrics['contrastive_loss'], on_epoch=True, prog_bar=True,
                     logger=True)
            losses.append(contrastive_loss)

        loss = sum(losses)
        self.log(f'{stage}_total_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'log': metrics}

    def _all_gather_proj(self, text_latents, image_latents, sync_grads=False):
        text_latents = self.all_gather(text_latents, sync_grads=sync_grads)
        image_latents = self.all_gather(image_latents, sync_grads=sync_grads)
        if isinstance(image_latents, list):
            image_latents = torch.cat(image_latents)
        elif len(image_latents.shape) == 3:
            image_latents = torch.cat([*image_latents])
        if isinstance(text_latents, list):
            text_latents = torch.cat(text_latents)
        elif len(text_latents.shape) == 3:
            text_latents = torch.cat([*text_latents])
        return text_latents, image_latents


class InverseAttentionTrainer(BaseTrainer):

    def vqa_step(self, vqa_images, vqa_input_ids, labels, targets):
        images = vqa_images.type(torch.float32)
        input_ids = vqa_input_ids.type(torch.long)
        labels = labels.type(torch.float32)
        loss_mask = torch.tensor(labels.clone().detach() == 2, dtype=torch.uint8)
        lm_logits = self.model('vqa', images=images, tokens=input_ids)
        labels = input_ids
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
        vqa_loss = self.vqa_criterion(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
        return vqa_loss

    def detection_step(self, detection_images, detection_input_ids, detection_attention_masks, boxes):
        images = detection_images.to(self.device, dtype=torch.float32)
        input_ids = detection_input_ids.to(self.device, dtype=torch.long)
        boxes = [boxes_per_label.to(self.device, dtype=torch.float) for boxes_per_label in boxes]
        detection_outputs = self.model('detection', images=images, tokens=input_ids)
        detection_loss = self.detection_criterion(detection_outputs, boxes)
        return detection_loss

    def model_step(self, batch, stage):
        (htr_images, encoded, encoded_length, gt_texts), (code_input_ids, code_input_labels, code_targets), (
        vqa_images, vqa_input_ids, labels, targets), (
        detection_names, detection_images, detection_input_ids, detection_attention_masks, boxes, size) = batch
        losses = []
        metrics = {}
        if len(htr_images) > 0:
            htr_loss = self.htr_step(htr_images, encoded, encoded_length)
            metrics['h_loss'] = htr_loss.detach().cpu().item()
            self.log(f'{stage}_handwritten_loss', metrics['h_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(htr_loss)

        if len(code_input_ids) > 0:
            c2c_loss = self.trans_step(code_input_ids, code_input_labels)
            metrics['c2c_loss'] = c2c_loss.detach().cpu().item()
            self.log(f'{stage}_c2c_loss', metrics['c2c_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(c2c_loss)

        if len(labels) > 0:
            vqa_loss = self.vqa_step(vqa_images, vqa_input_ids, labels, targets)
            metrics['vqa_loss'] = vqa_loss.detach().cpu().item()
            self.log(f'{stage}_vqa_loss', metrics['vqa_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(vqa_loss)

        if len(boxes) > 0:
            detection_loss = self.detection_step(
                detection_images, detection_input_ids, detection_attention_masks, boxes
            )
            sum_detection_losses = sum([
                loss * weight for loss, weight in zip(detection_loss.values(), self.detection_losses_weights)
            ])
            self.log(f'{stage}_detection_loss', sum_detection_losses, on_epoch=True, prog_bar=True, logger=True)
            for key in detection_loss:
                metrics[key] = detection_loss[key].detach().cpu().item()
                self.log(f'{stage}_{key}', metrics[key], on_epoch=True, prog_bar=True, logger=True)
            losses.append(sum_detection_losses)

        loss = sum(losses)
        self.log(f'{stage}_total_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'log': metrics}
