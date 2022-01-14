import comet_ml
import os
import json
import random
import argparse

import pandas as pd
import albumentations as A
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import SequentialSampler
import pytorch_lightning as pl
from catalyst.data import BalanceClassSampler, DistributedSamplerWrapper
from transformers import GPT2Model, GPT2Tokenizer

from fb_utils.utils import simple_detect_lang
from model.utils.utils import CTCLabeling
from model.dataset.dataset import DatasetRetriever, fb_collate_fn
from model.model import CrossAttentionGPT2FusionBrain, InverseAttentionGPT2FusionBrain
from model.trainer import CrossAttentionTrainer, InverseAttentionTrainer


if __name__ == '__main__':
    # Подготовка данных и сбор в единый DataFrame
    json_marking = json.load(open('handwritten/train_labels.json', 'rb'))
    marking = []
    for image_name, text in json_marking.items():
        marking.append({
            'path': image_name,
            'text': text,
            'lang': simple_detect_lang(text),
        })
    df_handwritten = pd.DataFrame(marking)
    df_handwritten['stage'] = 'train'
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    train_index, valid_index = next(skf.split(df_handwritten.index, df_handwritten['lang']))
    df_handwritten.loc[valid_index, 'stage'] = 'valid'
    # #
    # Detection
    # #
    json_true_zsod = json.load(open('russian_detection_vqa/vg_intersection_eng.json', 'rb'))
    json_true_zsod.update(json.load(open('russian_detection_vqa/vg_intersection_rus.json', 'rb')))
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
    json_questions = json.load(open('russian_detection_vqa/VQA_eng/vqa_questions_eng.json', 'rb'))
    json_questions.update(json.load(open('russian_detection_vqa/VQA_rus/vqa_questions_rus.json', 'rb')))
    json_answers = json.load(open('russian_detection_vqa/VQA_eng/vqa_answers_eng.json', 'rb'))
    json_answers.update(json.load(open('russian_detection_vqa/VQA_rus/vqa_answers_rus.json', 'rb')))
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
    # C2C
    # #
    df_c2c = pd.read_json(path_or_buf='c2c/java-python.jsonl', lines=True)
    train, test = train_test_split(df_c2c, test_size=0.2)
    valid, test = train_test_split(test, test_size=0.05)

    df_c2c.loc[train.index.to_list(), 'stage'] = 'train'
    df_c2c.loc[valid.index.to_list(), 'stage'] = 'valid'
    df_c2c.loc[test.index.to_list(), 'stage'] = 'test'

    # #
    # Merge in common set
    # #
    dataset = []
    for image_name, text, stage in zip(df_handwritten['path'], df_handwritten['text'], df_handwritten['stage']):
        dataset.append({
            'task_id': 'handwritten',
            'modality': 'image',
            'input_image': image_name,
            'output_text': text,
            'stage': stage,
        })

    for java, python, stage in zip(df_c2c['java'], df_c2c['python'], df_c2c['stage']):
        dataset.append({
            'task_id': 'trans',
            'modality': 'code',
            'input_text': java,
            'output_text': python,
            'stage': stage,
        })

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
        'handwritten': A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.25, always_apply=False),
            A.Rotate(limit=3, interpolation=1, border_mode=0, p=0.5),
            A.JpegCompression(quality_lower=75, p=0.5),
        ], p=1.0),
        'vqa': A.Compose([
            A.Resize(224, 224, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ], p=1.0),
        'detection': A.Compose([
            A.Resize(224, 224, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ], p=1.0)
    }

    # Подготовка предобученной модели и токенизатора, а также CTC Labeling для задачи распознавания рукописного текста
    CHARS = ' !"#&\'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
            '[]_abcdefghijklmnopqrstuvwxyz|}ЁАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№'
    ctc_labeling = CTCLabeling(CHARS)
    model_name = '/home/jovyan/vladimir/fb_baseline/gpt3_medium_py'
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

    attention_config = {
        'num_attention_layers': 2,
        'num_heads': 8,
        'pf_dim': 2048,
        'dropout': 0.1,
    }

    vqa_config = {
        'tokens_num': len(gpt_tokenizer),
    }

    detection_config = {
        'num_mlp_layers': 1,
        'num_queries': 8
    }

    model = CrossAttentionGPT2FusionBrain(
        gpt_model,
        attention_config=attention_config,
        handwritten_config=handwritten_config,
        vqa_config=vqa_config,
        detection_config=detection_config
    ) # InverseAttentionGPT2FusionBrain

    WORLD_SIZE = 8
    BATCH_SIZE = 16

    df_train = df[df['stage'] == 'train']
    df_valid = df[df['stage'] == 'valid']

    train_dataset = DatasetRetriever(
        task_ids=df_train['task_id'].values,
        input_images=df_train['input_image'].values,
        input_texts=df_train['input_text'].values,
        output_texts=df_train['output_text'].values,
        output_boxes=df_train['output_boxes'].values,
        ctc_labeling=ctc_labeling,
        tokenizer=gpt_tokenizer,
        stage='train',
        max_request_tokens_length=21,
        vqa_max_tokens_length=21,
        task_augs=task_augs,
    )
    valid_dataset = DatasetRetriever(
        task_ids=df_valid['task_id'].values,
        input_images=df_valid['input_image'].values,
        input_texts=df_valid['input_text'].values,
        output_texts=df_valid['output_text'].values,
        output_boxes=df_valid['output_boxes'].values,
        ctc_labeling=ctc_labeling,
        tokenizer=gpt_tokenizer,
        stage='valid',
        max_request_tokens_length=21,
        vqa_max_tokens_length=21,
        task_augs=task_augs,
    )

    CONFIG = {
        'lr': 0.000008,
        'max_lr': 0.0001,
        'pct_start': 0.1,
        'final_div_factor': 1000,
        'total_steps': 2000000 // WORLD_SIZE,
        'weights_path': None,
        'detection_losses': ['boxes', 'classification'],
        'detection_losses_weights': [1., 1., 1.],
        'temperature': 4.6052,
        'ignore_index': gpt_tokenizer.pad_token_id,
        'freeze': {
            'freeze_pos': False,
            'freeze_ln': False,
            'freeze_attn': False,
            'freeze_ff': False,
            'freeze_other': False
        }
    }

    lightning_model = CrossAttentionTrainer(model, CONFIG, ctc_labeling) # InverseAttentionTrainer(model, CONFIG, ctc_labeling)

    comet_logger = pl.loggers.CometLogger(
        api_key="8uucVJ9Hf7WVJuHwVz8DIA04H",
        workspace=os.environ.get("COMET_WORKSPACE"),
        save_dir="./LightningExperimentsNew/main_concat/logs",
        project_name="fusion_brain",
        experiment_name="fusion_concat_1",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_total_loss",
        dirpath="LightningExperimentsNew/main_concat/checkpoints/",
        filename="weight-{epoch:02d}",
        every_n_train_steps=20000,
        save_top_k=3,
        mode="min",
    )
    accelerator = 'dp' # 'ddp'
    trainer = pl.Trainer(gpus=-1, accelerator=accelerator, max_steps=CONFIG['total_steps'], check_val_every_n_epoch=1,
                         replace_sampler_ddp=True, default_root_dir='LightningExperimentsNew/main_concat/checkpoints/',
                         logger=comet_logger, callbacks=[lr_monitor, checkpoint_callback], num_sanity_val_steps=0)

    train_sampler = DistributedSamplerWrapper(
        sampler=BalanceClassSampler(labels=train_dataset.get_task_labels()),
        num_replicas=WORLD_SIZE,
        rank=trainer.global_rank,
        shuffle=True
    ) # SequentialSampler(train_dataset)

    valid_sampler = DistributedSamplerWrapper(
        sampler=BalanceClassSampler(labels=valid_dataset.get_task_labels()),
        num_replicas=WORLD_SIZE,
        rank=trainer.global_rank,
        shuffle=True
    ) # SequentialSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=8,
        collate_fn=fb_collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=8,
        collate_fn=fb_collate_fn,
    )

    trainer.fit(lightning_model, train_loader, valid_loader)
