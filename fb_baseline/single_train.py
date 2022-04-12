import comet_ml
import os
import json

import pandas as pd
import albumentations as A
from sklearn.model_selection import StratifiedKFold, train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import GPT2Model, GPT2Tokenizer

from .model.utils.utils import CTCLabeling
from .model.model import InverseAttentionGPT2FusionBrain
from .model.trainer import InverseAttentionTrainer
from fb_baseline.model.dataset.dataset import DetectionDataset, fusion_collate_fn


def run_train(conf):
    # #
    # Detection
    # #
    with open(conf.data.detection.requests, 'rb') as f:
        json_true_zsod = json.load(f)
    marking = []
    for image_name in json_true_zsod:
        for request, boxes in json_true_zsod[image_name].items():
            marking.append({
                'task_ids': 'detection',
                'images': os.path.join(conf.data.detection.images, image_name),
                'requests': request,
                'boxes': boxes,
            })
    df_detection = pd.DataFrame(marking)
    train_index, valid_index = train_test_split(df_detection, test_size=0.15)
    df_detection_train = df_detection.loc[train_index.index.to_list()]
    df_detection_train.index = pd.Index(range(df_detection_train.shape[0]))
    df_detection_valid = df_detection.loc[valid_index.index.to_list()]
    df_detection_valid.index = pd.Index(range(df_detection_valid.shape[0]))

    task_augs = {
        'handwritten': A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.25, always_apply=False),
            A.Rotate(limit=3, interpolation=1, border_mode=0, p=0.5),
            A.JpegCompression(quality_lower=75, p=0.5),
        ], p=1.0),
        'vqa': A.Compose([
            A.Resize(conf.common.image_size, conf.common.image_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ], p=1.0),
        'detection': A.Compose([
            A.Resize(conf.common.image_size, conf.common.image_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ], p=1.0)
    }

    with open(conf.model.ctc_labeling_chars) as f:
        ctc_labeling = CTCLabeling(f.read())
    model_name = conf.model.gpt_model
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token='<|pad|>')

    gpt_model = GPT2Model.from_pretrained(model_name)
    gpt_model.resize_token_embeddings(len(gpt_tokenizer))

    train_dataset = DetectionDataset(df_detection_train, gpt_tokenizer, conf.data.detection.max_request_tokens_length, 'train', task_augs)
    valid_dataset = DetectionDataset(df_detection_valid, gpt_tokenizer, conf.data.detection.max_request_tokens_length, 'valid', task_augs)
    # #
    # MODEL
    # #
    model = InverseAttentionGPT2FusionBrain(
        gpt_model,
        handwritten_config=conf.model.handwritten,
        vqa_config={'tokens_num': len(gpt_tokenizer)},
        detection_config=conf.model.detection
    )

    lightning_model = InverseAttentionTrainer(model, conf.trainer, gpt_tokenizer, ctc_labeling)

    comet_logger = pl.loggers.CometLogger(
        api_key=None,
        workspace=os.environ.get("COMET_WORKSPACE"),
        **conf.logger.commet
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(**conf.model_checkpoint)
    ]

    trainer = pl.Trainer(gpus=conf.trainer.gpus, strategy=conf.trainer.strategy, max_steps=conf.trainer.total_steps,
                         check_val_every_n_epoch=1, replace_sampler_ddp=True, default_root_dir=conf.model_checkpoint.dirpath,
                         logger=comet_logger, callbacks=callbacks, num_sanity_val_steps=0, accumulate_grad_batches=conf.trainer.accumulate_grad_batches)

    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.data.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=conf.data.num_workers,
        collate_fn=fusion_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=conf.data.batch_size,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=conf.data.num_workers,
        collate_fn=fusion_collate_fn,
    )

    trainer.fit(lightning_model, train_loader, valid_loader)
