import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..utils.utils import resize_if_need, make_img_padding


class HTRDataset(Dataset):

    def __init__(self, dataframe, ctc_labeling, image_w, image_h, task_augs=None):
        super().__init__()
        self.task_ids = dataframe["task_ids"].copy()
        self.images = dataframe["images"].copy()
        self.gt_texts = dataframe["gt_texts"].copy()

        self.task_augs = task_augs or {}
        self.ctc_labeling = ctc_labeling

        self.image_w = image_w
        self.image_h = image_h

    def change_index(self, index_shift):
        self.task_ids.index += index_shift
        self.images.index += index_shift
        self.gt_texts.index += index_shift

    def __len__(self):
        return self.task_ids.shape[0]

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _ = resize_if_need(image, self.image_h, self.image_w)
        image = make_img_padding(image, self.image_h, self.image_w)

        transforms = self.task_augs.get('handwritten')
        if transforms:
            image = transforms(image=image)['image']
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        gt_text = self.gt_texts[idx]
        encoded = self.ctc_labeling.encode(gt_text)
        encoded = torch.tensor(encoded, dtype=torch.long)

        return {
            'task_id': self.task_ids[idx],
            'image': image,
            'gt_text': gt_text,
            'encoded': encoded,
        }


class C2CDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_in_code_length, max_out_code_length, stage):
        super().__init__()
        self.task_ids = dataframe["task_ids"].copy()
        self.input_code = dataframe["java"].copy()
        self.output_code = dataframe["python"].copy()

        self.tokenizer = tokenizer
        self.stage = stage

        self.max_in_code_length = max_in_code_length
        self.max_out_code_length = max_out_code_length

    def change_index(self, index_shift):
        self.task_ids.index += index_shift
        self.input_code.index += index_shift
        self.output_code.index += index_shift

    def __len__(self):
        return self.task_ids.shape[0]

    def __getitem__(self, idx):
        source = "Java: " + self.input_code[idx] + " Python: "
        encoded_source = self.tokenizer.encode(source)
        target = self.output_code[idx]
        encoded_target = self.tokenizer.encode(target)

        if self.stage == 'train' or self.stage == 'valid':
            input_tokens = encoded_source[:self.max_in_code_length]
            output_tokens = encoded_target[:self.max_out_code_length]
            input_ids = input_tokens + output_tokens + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)

            pad_len = self.max_in_code_length + self.max_out_code_length + 1 - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            input_ids = encoded_source[:self.max_in_code_length]
            attention_mask = [1] * len(input_ids)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            'task_id': self.task_ids[idx],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': target
        }


class VQADataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_question_tokens_length, max_answer_tokens_length, stage, task_augs=None):
        super().__init__()
        self.task_ids = dataframe["task_ids"].copy()
        self.images = dataframe["images"].copy()
        self.questions = dataframe["questions"].copy()
        self.answers = dataframe["answers"].copy()

        self.task_augs = task_augs or {}
        self.tokenizer = tokenizer
        self.stage = stage

        self.max_question_tokens_length = max_question_tokens_length
        self.max_answer_tokens_length = max_answer_tokens_length

    def __len__(self):
        return self.task_ids.shape[0]

    def change_index(self, index_shift):
        self.task_ids.index += index_shift
        self.images.index += index_shift
        self.questions.index += index_shift
        self.answers.index += index_shift

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w, _ = image.shape

        transforms = self.task_augs.get('detection')
        if transforms:
            image = transforms(image=image)['image']
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        question = "Question: " + self.questions[idx] + " Answer: "
        input_tokens = self.tokenizer.encode(question)
        answer = self.answers[idx]
        if self.stage == 'train' or self.stage == 'valid':
            answer += "."
            output_tokens = self.tokenizer.encode(answer)
            input_tokens = input_tokens[:self.max_question_tokens_length]
            output_tokens = output_tokens[:self.max_answer_tokens_length]
            input_ids = input_tokens + output_tokens
            attention_mask = [1] * len(input_ids)

            pad_len = self.max_question_tokens_length + self.max_answer_tokens_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            input_ids = input_tokens[:self.max_question_tokens_length]
            attention_mask = [1] * len(input_ids)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            'task_id': self.task_ids[idx],
            'image_name': os.path.basename(path),
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': answer
        }


class DetectionDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_request_tokens_length, stage, task_augs=None):
        super().__init__()
        self.task_ids = dataframe["task_ids"].copy()
        self.images = dataframe["images"].copy()
        self.requests = dataframe["requests"].copy()
        self.boxes = dataframe["boxes"].copy()

        self.task_augs = task_augs or {}
        self.tokenizer = tokenizer
        self.stage = stage

        self.max_request_tokens_length = max_request_tokens_length

    def change_index(self, index_shift):
        self.task_ids.index += index_shift
        self.images.index += index_shift
        self.requests.index += index_shift
        self.boxes.index += index_shift

    def __len__(self):
        return self.task_ids.shape[0]

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w, _ = image.shape

        transforms = self.task_augs.get('detection')
        if transforms:
            image = transforms(image=image)['image']
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        request = self.requests[idx]
        boxes = self.boxes[idx]
        if self.stage == 'train' or self.stage == 'valid':
            request = 'Request: ' + request + '.'
            input_tokens = self.tokenizer.encode(request)
            input_ids = input_tokens[:self.max_request_tokens_length]
            attention_mask = [1] * len(input_ids)

            pad_len = self.max_request_tokens_length - len(input_ids)
            input_ids += [self.tokenizer.eos_token_id] * pad_len
            attention_mask += [0] * pad_len
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, [0, 2]] /= image_w
            boxes[:, [1, 3]] /= image_h
        else:
            input_ids = list(map(lambda x: self.tokenizer.encode('Request: ' + x + '.'), request))
            attention_mask = [torch.ones(len(tokens), dtype=torch.long) for tokens in input_ids]
            input_ids = [torch.tensor(input_id, dtype=torch.long) for input_id in input_ids]

        return {
            'task_id': self.task_ids[idx],
            'image_name': os.path.basename(path),
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'boxes': boxes,
            'size': (image_h, image_w)
        }


class FusionDataset(Dataset):

    def __init__(self, single_datasets, weights):
        super().__init__()
        self.single_datasets = single_datasets

        index_shifts = np.cumsum([0] + [len(dataset) for dataset in single_datasets.values()])
        for index_shift, dataset_name in zip(index_shifts[:-1], single_datasets.keys()):
            self.single_datasets[dataset_name].change_index(index_shift)
        self.task_ids = pd.concat([dataset.task_ids for dataset in single_datasets.values()])

        self.weights = []
        for dataset_name in single_datasets.keys():
            self.weights += [weights[dataset_name] / len(single_datasets[dataset_name])] * len(single_datasets[dataset_name])

    def __getitem__(self, idx):
        task_id = self.task_ids[idx]
        sample = self.single_datasets.get(task_id, {'task_id': task_id})[idx]
        return sample

    def __len__(self):
        return self.task_ids.shape[0]


def htr_collate_fn(batch):
    encoded, encoded_length, images, gt_texts = [], [], [], []
    for sample in batch:
        encoded.append(sample['encoded'])
        encoded_length.append(sample['encoded'].shape[0])
        images.append(sample['image'])
        gt_texts.append(sample['gt_text'])
    if batch:
        images = pad_sequence(images, batch_first=True)
        encoded = pad_sequence(encoded, batch_first=True)
        encoded_length = torch.tensor(encoded_length)
    return images, encoded, encoded_length, gt_texts


def c2c_collate_fn(batch):
    input_ids, attention_masks, targets = [], [], []
    for sample in batch:
        input_ids.append(sample['input_ids'])
        attention_masks.append(sample['attention_mask'])
        targets.append(sample['target'])
    if batch:
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
    return input_ids, attention_masks, targets


def vqa_collate_fn(batch):
    images, input_ids, attention_masks, targets = [], [], [], []
    for sample in batch:
        images.append(sample['image'])
        input_ids.append(sample['input_ids'])
        attention_masks.append(sample['attention_mask'])
        targets.append(sample['target'])
    if batch:
        images = torch.stack(images)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
    return images, input_ids, attention_masks, targets


def detection_collate_fn(batch):
    names, images, input_ids, attention_masks, boxes, size = [], [], [], [], [], []
    for sample in batch:
        images.append(sample['image'])
        input_ids.append(sample['input_ids'])
        attention_masks.append(sample['attention_mask'])
        boxes.append(sample['boxes'])
        size.append(sample['size'])
        names.append(sample['image_name'])
    if batch:
        images = torch.stack(images)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
    return names, images, input_ids, attention_masks, boxes, size


def fusion_collate_fn(batch):
    single_collate_fn = {
        "handwritten": htr_collate_fn,
        "c2c": c2c_collate_fn,
        "vqa": vqa_collate_fn,
        "detection": detection_collate_fn,
    }
    single_batch = {
        "handwritten": [],
        "c2c": [],
        "vqa": [],
        "detection": [],
    }
    for sample in batch:
        single_batch[sample['task_id']].append(sample)
    return [single_collate_fn[key](single_batch[key]) for key in single_collate_fn.keys()]
