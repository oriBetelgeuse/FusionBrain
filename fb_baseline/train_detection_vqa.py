import comet_ml
import os
import json
import random

import pandas as pd
import albumentations as A
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import pytorch_lightning as pl
from catalyst.data import BalanceClassSampler, DistributedSamplerWrapper
from transformers import GPT2Model, GPT2Tokenizer

from .custom_horovod_plugin import CustomHorovodPlugin
from .fb_utils.utils import simple_detect_lang
from .model.utils.utils import CTCLabeling
from .model.dataset.dataset import DatasetRetriever, fb_collate_fn
from .model.model import InverseAttentionGPT2FusionBrain
from .model.trainer import InverseAttentionTrainer


def run_train(conf):
    # #
    # Detection
    # #
    json_true_zsod = json.load(open(conf.data.detection.requests, 'rb'))
    marking = []
    for image_name in json_true_zsod:
        marking.extend([{
            'task_id': 'detection',
            'path': image_name,
            'req': request,
            'boxes': boxes,
            'lang': simple_detect_lang(request)
        } for request, boxes in json_true_zsod[image_name].items() if boxes])
    df_detection = pd.DataFrame(marking)
    df_detection['stage'] = 'train'
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    train_index, valid_index = next(skf.split(df_detection.index, df_detection['lang']))
    df_detection.loc[valid_index, 'stage'] = 'valid'
    # #
    # VQA
    # #
    json_questions = json.load(open(conf.data.vqa.questions, 'rb'))
    json_answers = json.load(open(conf.data.vqa.answers, 'rb'))
    marking = []
    for key in json_questions:
        marking.extend([{
            'path': str(json_questions[key]['image_id']) + '.jpg',
            'question': json_questions[key]['question'],
            'answer': answer,
            'lang': simple_detect_lang(answer)
        } for answer in json_answers[key]['answer']])
    df_vqa = pd.DataFrame(marking)
    df_vqa['stage'] = 'train'
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    train_index, valid_index = next(skf.split(df_vqa.index, df_vqa['lang']))
    df_vqa.loc[valid_index, 'stage'] = 'valid'
    # #
    # Merge in common set
    # #
    dataset = []
    for image_name, text_input, text_output, stage in zip(df_vqa['path'], df_vqa['question'], df_vqa['answer'],
                                                          df_vqa['stage']):
        dataset.append({
            'task_id': 'vqa',
            'modality': 'image+text',
            'input_image': image_name,
            'input_text': text_input,
            'output_text': text_output,
            'stage': stage,
        })
    for image_name, text_input, boxes, stage in zip(df_detection['path'], df_detection['req'], df_detection['boxes'],
                                                    df_detection['stage']):
        dataset.append({
            'task_id': 'detection',
            'modality': 'image+text',
            'input_image': image_name,
            'input_text': text_input,
            'output_boxes': boxes,
            'stage': stage,
        })

    random.shuffle(dataset)
    df = pd.DataFrame(dataset)

    task_augs = {
        'vqa': A.Compose([
            A.Resize(conf.common.image_size, conf.common.image_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ], p=1.0),
        'detection': A.Compose([
            A.Resize(conf.common.image_size, conf.common.image_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ], p=1.0)
    }

    CHARS = ' !"#&\'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
            '[]_abcdefghijklmnopqrstuvwxyz|}ЁАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№'
    ctc_labeling = CTCLabeling(CHARS)
    model_name = conf.model.gpt_model
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<s>',
                                                  eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                                  sep_token='<|SEP|>')

    gpt_model = GPT2Model.from_pretrained(model_name)
    gpt_model.resize_token_embeddings(len(gpt_tokenizer))

    handwritten_config = {
        'patch_w': 8,
        'patch_h': 128,
        'in_layer_sizes': [8 * 128 * 3],
        'out_layer_sizes': [64],
        'orth_gain': 1.41,
        'dropout': 0.1,
        'lstm_num_layers': 3,
        'output_dim': len(ctc_labeling),  # 152
    }

    model = InverseAttentionGPT2FusionBrain(
        gpt_model,
        handwritten_config=handwritten_config,
        vqa_config={'tokens_num': len(gpt_tokenizer)},
        detection_config=conf.model.detection
    )

    df_train = df[df['stage'] == 'train']
    df_valid = df[df['stage'] == 'valid']

    train_dataset = DatasetRetriever(
        handwritten_images=None,
        detection_images=conf.data.detection.images,
        vqa_images=conf.data.vqa.images,
        task_ids=df_train['task_id'].values,
        input_images=df_train['input_image'].values,
        input_texts=df_train['input_text'].values,
        output_texts=df_train['output_text'].values,
        output_boxes=df_train['output_boxes'].values,
        ctc_labeling=ctc_labeling,
        tokenizer=gpt_tokenizer,
        stage='train',
        max_request_tokens_length=conf.data.detection.max_request_tokens_length,
        vqa_max_tokens_length=conf.data.vqa.max_vqa_tokens_length,
        task_augs=task_augs,
    )
    valid_dataset = DatasetRetriever(
        handwritten_images=None,
        detection_images=conf.data.detection.images,
        vqa_images=conf.data.vqa.images,
        task_ids=df_valid['task_id'].values,
        input_images=df_valid['input_image'].values,
        input_texts=df_valid['input_text'].values,
        output_texts=df_valid['output_text'].values,
        output_boxes=df_valid['output_boxes'].values,
        ctc_labeling=ctc_labeling,
        tokenizer=gpt_tokenizer,
        stage='valid',
        max_request_tokens_length=conf.data.detection.max_request_tokens_length,
        vqa_max_tokens_length=conf.data.vqa.max_vqa_tokens_length,
        task_augs=task_augs,
    )

    lightning_model = InverseAttentionTrainer(model, conf.trainer, gpt_tokenizer, ctc_labeling)

    comet_logger = pl.loggers.CometLogger(
        api_key="8uucVJ9Hf7WVJuHwVz8DIA04H",
        workspace=os.environ.get("COMET_WORKSPACE"),
        **conf.logger.commet
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(**conf.model_checkpoint)
    ]
    plugins = None
    if conf.trainer.accelerator == 'horovod':
        backward_passes_per_step = 1
        plugins = [CustomHorovodPlugin(backward_passes_per_step=backward_passes_per_step)]

    trainer = pl.Trainer(gpus=conf.trainer.gpus, accelerator=conf.trainer.accelerator, max_steps=conf.trainer.total_steps,
                         check_val_every_n_epoch=1, replace_sampler_ddp=True, default_root_dir=conf.model_checkpoint.dirpath,
                         plugins=plugins, logger=comet_logger, callbacks=callbacks, num_sanity_val_steps=0)

    train_sampler = DistributedSamplerWrapper(
        sampler=BalanceClassSampler(labels=train_dataset.get_task_labels()),
        num_replicas=conf.data.world_size,
        rank=trainer.global_rank,
        shuffle=True
    )

    valid_sampler = DistributedSamplerWrapper(
        sampler=BalanceClassSampler(labels=valid_dataset.get_task_labels()),
        num_replicas=conf.data.world_size,
        rank=trainer.global_rank,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.data.batch_size,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=conf.data.num_workers,
        collate_fn=fb_collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=conf.data.batch_size,
        sampler=valid_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=conf.data.num_workers,
        collate_fn=fb_collate_fn,
    )

    trainer.fit(lightning_model, train_loader, valid_loader)
