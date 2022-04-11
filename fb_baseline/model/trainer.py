from collections import OrderedDict

import torch
from torch import nn
import pytorch_lightning as pl

from .modules.loss import DetectionCriterion


class InverseAttentionTrainer(pl.LightningModule):

    def __init__(self, model, config, gpt_tokenizer, ctc_labeling):
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
        self.c2c_criterion = nn.CrossEntropyLoss(ignore_index=gpt_tokenizer.pad_token_id)
        self.vqa_criterion = nn.CrossEntropyLoss(ignore_index=gpt_tokenizer.pad_token_id)
        self.detection_criterion = DetectionCriterion(config['detection_losses'])
        self.detection_losses_weights = config['detection_losses_weights']

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

    def htr_step(self, images, encoded, encoded_length):
        bs = images.shape[0]
        handwritten_outputs = self.model('handwritten', images=images)
        preds_size = torch.IntTensor([handwritten_outputs.size(1)] * bs)
        preds = handwritten_outputs.log_softmax(2).permute(1, 0, 2)
        handwritten_loss = self.handwritten_criterion(preds, encoded, preds_size, encoded_length)
        return handwritten_loss

    def trans_step(self, input_ids, attention_masks):
        lm_logits = self.model('trans', tokens=input_ids, attention_masks=attention_masks)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        c2c_loss = self.c2c_criterion(shift_logits.transpose(-1, -2), shift_labels)
        return c2c_loss

    def vqa_step(self, images, input_ids, attention_masks):
        lm_logits = self.model('vqa', images=images, tokens=input_ids, attention_masks=attention_masks)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        vqa_loss = self.vqa_criterion(shift_logits.transpose(-1, -2), shift_labels)
        return vqa_loss

    def detection_step(self, images, input_ids, attention_masks, boxes):
        detection_outputs = self.model('detection', images=images, tokens=input_ids, attention_masks=attention_masks)
        detection_loss = self.detection_criterion(detection_outputs, boxes)
        return detection_loss

    def model_step(self, batch, stage):
        (htr_images, encoded, encoded_length, _),\
        (code_input_ids, code_attention_masks, _),\
        (vqa_images, vqa_input_ids, vqa_attention_masks, _),\
        (_, detection_images, detection_input_ids, detection_attention_masks, boxes, _) = batch
        losses = []
        metrics = {}
        if len(htr_images) > 0:
            htr_loss = self.htr_step(htr_images, encoded, encoded_length)
            metrics['h_loss'] = htr_loss.detach().cpu().item()
            self.log(f'{stage}_handwritten_loss', metrics['h_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(htr_loss)

        if len(code_input_ids) > 0:
            c2c_loss = self.trans_step(code_input_ids, code_attention_masks)
            metrics['c2c_loss'] = c2c_loss.detach().cpu().item()
            self.log(f'{stage}_c2c_loss', metrics['c2c_loss'], on_epoch=True, prog_bar=True, logger=True)
            losses.append(c2c_loss)

        if len(vqa_attention_masks) > 0:
            vqa_loss = self.vqa_step(vqa_images, vqa_input_ids, vqa_attention_masks)
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
        metrics['total_loss'] = loss.detach().cpu().item()
        self.log(f'{stage}_total_loss', metrics['total_loss'], on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'log': metrics}

    def training_step(self, train_batch, batch_idx):
        return self.model_step(train_batch, 'train')

    def validation_step(self, valid_batch, batch_idx):
        self.model_step(valid_batch, 'valid')
