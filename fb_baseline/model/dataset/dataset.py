import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..utils.utils import resize_if_need, make_img_padding


class DatasetRetriever(Dataset):

    def __init__(self,
                 task_ids,
                 input_images,
                 input_texts,
                 output_texts,
                 output_boxes,
                 ctc_labeling,
                 tokenizer,
                 stage,
                 max_request_tokens_length,
                 vqa_max_tokens_length,
                 task_augs=None):
        super().__init__()
        self.task_ids = task_ids

        self.input_images = input_images
        self.input_texts = input_texts
        self.output_texts = output_texts

        self.task_augs = task_augs or {}
        self.tokenizer = tokenizer
        self.stage = stage

        # handwritten[image]:
        self.ctc_labeling = ctc_labeling
        self.handwritten_image_w = 512
        self.handwritten_image_h = 128

        # code2code
        self.code_max_length = 512

        # detection[image, text]:
        self.max_request_tokens_length = max_request_tokens_length
        self.output_boxes = output_boxes

        # vqa[image, text]:
        self.vqa_max_tokens_length = vqa_max_tokens_length

    def __getitem__(self, idx):
        task_id = self.task_ids[idx]
        if task_id == 'handwritten':
            return self.get_handwritten_sample(idx)
        elif task_id == 'trans':
            return self.get_trans_sample(idx)
        elif task_id == 'detection':
            return self.get_detection_sample(idx)
        elif task_id == 'vqa':
            return self.get_vqa_sample(idx)
        return {'task_id': task_id}

    def get_trans_sample(self, idx):

        source = self.input_texts[idx]
        encoded_source = self.tokenizer.encode(str(source))
        target = self.output_texts[idx]
        encoded_target = self.tokenizer.encode(str(target))

        input_ids, input_labels = self.pad_and_get_mask(encoded_target, encoded_source, self.tokenizer)
        input_ids, input_labels = torch.tensor(input_ids), torch.tensor(input_labels)

        return {
            'task_id': self.task_ids[idx],
            'input_ids': input_ids,
            'input_labels': input_labels,
            'target': target
        }

    def get_handwritten_sample(self, idx):
        path = 'handwritten/images/' + self.input_images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _ = self.resize_image(image)

        gt_text = self.output_texts[idx]
        encoded = self.ctc_labeling.encode(gt_text)

        ## Augs ##
        transforms = self.task_augs.get('handwritten')
        if transforms:
            image = transforms(image=image)['image']
        ##########

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            'task_id': self.task_ids[idx],
            'image': image,
            'gt_text': gt_text,
            'encoded': torch.tensor(encoded, dtype=torch.int32),
        }

    def get_detection_sample(self, idx):
        path = 'russian_detection_vqa/images/' + self.input_images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w, _ = image.shape

        ## Augs ##
        transforms = self.task_augs.get('detection')
        if transforms:
            image = transforms(image=image)['image']
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image_name = self.input_images[idx]
        ##########

        ## Input tokens and Boxes##
        output_boxes = self.output_boxes[idx]
        if self.stage == 'train' or self.stage == 'valid':
            input_text = self.input_texts[idx]
            input_tokens = self.tokenizer.encode_plus(input_text)
            input_tokens['input_ids'] = input_tokens['input_ids'][:21]
            input_tokens['attention_mask'] = input_tokens['attention_mask'][:21]
            pad_len = self.max_request_tokens_length - len(input_tokens['input_ids'])
            input_tokens['input_ids'] += [self.tokenizer.pad_token_id] * pad_len
            input_tokens['attention_mask'] += [0] * pad_len
            input_ids = torch.tensor(input_tokens['input_ids'])
            attention_mask = torch.tensor(input_tokens['attention_mask'])

            output_boxes = torch.tensor(output_boxes, dtype=torch.float32)
            output_boxes[:, 0] /= image_w
            output_boxes[:, 1] /= image_h
            output_boxes[:, 2] /= image_w
            output_boxes[:, 3] /= image_h
        else:
            input_texts = self.input_texts[idx].split(';')
            input_ids = list(map(self.tokenizer.encode, input_texts))
            attention_mask = [[1 for _ in input_token] for input_token in input_ids]
            input_ids = [torch.tensor(input_id) for input_id in input_ids]
            attention_mask = [torch.tensor(mask) for mask in attention_mask]

            output_boxes = {
                input_text: boxes for input_text, boxes in zip(input_texts, output_boxes)
            }
        ###########

        return {
            'task_id': self.task_ids[idx],
            'image_name': image_name,
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'boxes': output_boxes,
            'size': (image_h, image_w)
        }

    def get_vqa_sample(self, idx):
        path = 'russian_detection_vqa/images/' + self.input_images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## Augs ##
        transforms = self.task_augs.get('vqa')
        if transforms:
            image = transforms(image=image)['image']
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image_name = self.input_images[idx]
        ##########

        ## Question and Answer ##
        input_text = self.input_texts[idx]
        input_tokens = self.tokenizer.encode(input_text)
        output_text = self.output_texts[idx]
        output_tokens = self.tokenizer.encode(output_text)

        if self.stage == 'train' or self.stage == 'valid':
            input_tokens, output_tokens = input_tokens[:12], output_tokens[:7]
            input_ids = input_tokens + [self.tokenizer.bos_token_id] + output_tokens + [self.tokenizer.eos_token_id]
            labels = [1] * len(input_tokens) + [2] * (len(output_tokens) + 1) + [0]

            pad_len = self.vqa_max_tokens_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [0] * pad_len
        else:
            input_ids = input_tokens + [self.tokenizer.bos_token_id]
            labels = [1] * len(input_tokens) + [2]
        ##########

        return {
            'task_id': self.task_ids[idx],
            'image_name': image_name,
            'image': image,
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'target': output_text
        }

    def resize_image(self, image):
        image, coef = resize_if_need(image, self.handwritten_image_h, self.handwritten_image_w)
        image = make_img_padding(image, self.handwritten_image_h, self.handwritten_image_w)
        return image, coef

    def pad_and_get_mask(self, target, source, tokenizer):
        if self.stage == 'test':
            target = []
        while len(target) + len(source) + 2 > self.code_max_length:
            if len(target) > len(source):
                target = target[:-1]
            else:
                source = source[:-1]
        if self.stage == 'train' or self.stage == 'valid':
            inputs = source + [tokenizer.bos_token_id] + target + [tokenizer.eos_token_id]
            labels = [1] * len(source) + [2] * (len(target) + 1) + [0]

        else:
            inputs = source + [tokenizer.bos_token_id]
            labels = [1] * len(source) + [2]

            return inputs, labels

        assert len(inputs) <= self.code_max_length
        pad_len = self.code_max_length - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)

        return inputs, labels

    def __len__(self) -> int:
        return self.task_ids.shape[0]

    def get_task_labels(self):
        return list(self.task_ids)


def fb_collate_fn(batch):
    """ fusion brain collate fn """
    encoded, encoded_length, htr_images, gt_texts = [], [], [], []  # handwritten[image]
    code_input_ids, code_input_labels, code_targets = [], [], []  # code
    vqa_images, vqa_input_ids, labels, targets = [], [], [], []  # vqa[image, text]
    detection_names, detection_images, detection_input_ids, detection_attention_masks, boxes, size = [], [], [], [], [], []  # detection[image, text]

    for i, sample in enumerate(batch):
        if sample['task_id'] == 'handwritten':
            encoded.append(sample['encoded'])
            encoded_length.append(sample['encoded'].shape[0])
            htr_images.append(sample['image'])
            gt_texts.append(sample['gt_text'])
        elif sample['task_id'] == 'trans':
            code_input_ids.append(sample['input_ids'])
            code_input_labels.append(sample['input_labels'])
            code_targets.append(sample['target'])
        elif sample['task_id'] == 'detection':
            detection_images.append(sample['image'])
            detection_input_ids.append(sample['input_ids'])
            detection_attention_masks.append(sample['attention_mask'])
            boxes.append(sample['boxes'])
            size.append(sample['size'])
            detection_names.append(sample['image_name'])
        elif sample['task_id'] == 'vqa':
            vqa_images.append(sample['image'])
            vqa_input_ids.append(sample['input_ids'])
            labels.append(sample['labels'])
            targets.append(sample['target'])

    if htr_images:
        htr_images = pad_sequence(htr_images, batch_first=True)
        encoded, encoded_length = pad_sequence(encoded, batch_first=True), torch.tensor(encoded_length)
    if detection_images:
        detection_images = torch.stack(detection_images)
    if vqa_images:
        vqa_images = torch.stack(vqa_images)
    if detection_attention_masks and torch.is_tensor(detection_attention_masks[0]):
        detection_input_ids = pad_sequence(detection_input_ids, batch_first=True)
        detection_attention_masks = torch.stack(detection_attention_masks)
    elif detection_attention_masks:
        detection_input_ids = [input_id.unsqueeze(0) for input_id in detection_input_ids[0]]
        detection_attention_masks = [attention_mask.unsqueeze(0) for attention_mask in detection_attention_masks[0]]
    if labels:
        vqa_input_ids = pad_sequence(vqa_input_ids, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)
    if code_input_ids:
        code_input_ids = pad_sequence(code_input_ids, batch_first=True)
        code_input_labels = pad_sequence(code_input_labels, batch_first=True)
    return (htr_images, encoded, encoded_length, gt_texts), (code_input_ids, code_input_labels, code_targets), (
    vqa_images, vqa_input_ids, labels, targets), (
           detection_names, detection_images, detection_input_ids, detection_attention_masks, boxes, size)
