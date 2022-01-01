from abc import abstractmethod

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

from modules.layers import ResnetBackbone, CrossAttentionLayer, MLP


class BaseGPT2FusionBrain(nn.Module):

    def __init__(self, gpt_model, handwritten_config, **freeze_gpt_kwargs):
        super().__init__()
        self.gpt_model = gpt_model
        self.embedding_size = self.gpt_model.config.n_embd
        self.freeze_gpt(**freeze_gpt_kwargs)

        # handwritten[image] input/output layers:
        self.handwritten_config = handwritten_config
        self.handwritten_input_layer = self._build_input_net(
            input_dim=handwritten_config['patch_w'] * handwritten_config['patch_h'] * 3,
            in_layer_sizes=handwritten_config['in_layer_sizes'],
            orth_gain=handwritten_config['orth_gain'],
            dropout=handwritten_config['dropout'],
        )
        self.handwritten_lstm = nn.LSTM(
            self.embedding_size, self.embedding_size // 2,
            handwritten_config['lstm_num_layers'], dropout=handwritten_config['dropout'],
            batch_first=True, bidirectional=True
        )
        self.handwritten_output_layer = self._build_output_net(
            output_dim=handwritten_config['output_dim'],
            out_layer_sizes=handwritten_config['out_layer_sizes'],
            dropout=handwritten_config['dropout'],
        )
        print('=== HANDWRITTEN TASK ===')
        self._calculate_trainable_params([
            self.handwritten_input_layer,
            self.gpt_model,
            self.handwritten_lstm,
            self.handwritten_output_layer,
        ], without_emb=True)
        print('=== === === === ===')
        #####

        # code2code[code]
        self.beam_size = 3
        self.sos_id = self.gpt_model.config.bos_token_id
        self.eos_id = self.gpt_model.config.eos_token_id
        self.lm_head = nn.Linear(self.gpt_model.config.n_embd, self.gpt_model.config.vocab_size, bias=False)

        print('=== C2C TASK ===')
        self._calculate_trainable_params([self.gpt_model, self.lm_head])
        print('=== === === === ===')
        #####

    def forward(self, task_id, **kwargs):
        if task_id == 'handwritten':
            return self.forward_handwritten(**kwargs)
        elif task_id == 'trans':
            return self.forward_trans(**kwargs)
        elif task_id == 'vqa':
            return self.forward_vqa(**kwargs)
        elif task_id == 'detection':
            return self.forward_detection(**kwargs)

    def forward_trans(self, input_ids, input_labels=None, eval_bleu=False, past=None):
        if not eval_bleu:
            attn_mask = torch.tensor(input_labels.clone().detach() != 0, dtype=torch.uint8)
            attn_mask = attn_mask.to(input_labels.device)
            outputs = self.gpt_model(input_ids, attention_mask=attn_mask)
            x = self.lm_head(outputs[0])
            return x
        else:
            if past is not None:
                outputs = self.gpt_model(input_ids, past_key_values=past)
                logits = self.lm_head(outputs[0])
                return logits, outputs[1]
            else:
                outputs = self.gpt_model(input_ids)[1]
                return outputs

    def forward_handwritten(self, images):
        x = rearrange(images, 'b c (h p1) (w p2) -> b (w) (h) (p1 p2 c)',
                      p1=self.handwritten_config['patch_h'], p2=self.handwritten_config['patch_w'])
        x = x.squeeze(2)
        x = self.handwritten_input_layer(x)
        # Fusion Brain
        transformer_outputs = self.gpt_model(inputs_embeds=x, output_hidden_states=True)
        x = transformer_outputs.last_hidden_state
        #####
        x, _ = self.handwritten_lstm(x)
        x = self.handwritten_output_layer(x)
        return x

    @abstractmethod
    def forward_vqa(self, images, tokens):
        return

    @abstractmethod
    def forward_detection(self, images, tokens):
        return

    def freeze_gpt(self, freeze_pos=True, freeze_ln=True, freeze_attn=True, freeze_ff=True, freeze_other=True):
        for name, p in self.gpt_model.named_parameters():
            name = name.lower()
            if 'ln' in name or 'norm' in name:
                p.requires_grad = not freeze_ln
            elif 'wpe' in name or 'position_embeddings' in name or 'pos_drop' in name:
                p.requires_grad = not freeze_pos
            elif 'mlp' in name:
                p.requires_grad = not freeze_ff
            elif 'attn' in name:
                p.requires_grad = not freeze_attn
            else:
                p.requires_grad = not freeze_other

    def _build_input_net(self, input_dim, in_layer_sizes=None, orth_gain=1.41, dropout=0.1):
        """ вспомогательный метод для сборки input слоя, который приводит размер входящих данный к эмбеддингу gpt """
        in_layer_sizes = [] if not in_layer_sizes else in_layer_sizes
        in_layers = []
        last_output_size = input_dim
        for size in in_layer_sizes:
            layer = nn.Linear(last_output_size, size)
            if orth_gain is not None:
                torch.nn.init.orthogonal_(layer.weight, gain=orth_gain)
            layer.bias.data.zero_()

            in_layers.append(layer)
            in_layers.append(nn.ReLU())
            in_layers.append(nn.Dropout(dropout))
            last_output_size = size

        final_linear = nn.Linear(last_output_size, self.embedding_size)
        if orth_gain is not None:
            torch.nn.init.orthogonal_(final_linear.weight, gain=orth_gain)
        final_linear.bias.data.zero_()

        in_layers.append(final_linear)
        in_layers.append(nn.Dropout(dropout))

        return nn.Sequential(*in_layers)

    def _build_output_net(self, output_dim, embedding_size=None, out_layer_sizes=None, dropout=0.1):
        """ вспомогательный метод для сборки output слоя """
        out_layer_sizes = [] if not out_layer_sizes else out_layer_sizes
        out_layers = []
        last_output_size = embedding_size or self.embedding_size
        for size in out_layer_sizes:
            out_layers.append(nn.Linear(last_output_size, size))
            out_layers.append(nn.ReLU())
            out_layers.append(nn.Dropout(dropout))
            last_output_size = size
        out_layers.append(nn.Linear(last_output_size, output_dim))
        return nn.Sequential(*out_layers)

    def _calculate_trainable_params(self, layers, without_emb=False):
        trainable_params, all_used_params = 0, 0
        for layer in layers:
            if layer == self.gpt_model and without_emb:
                layer_parameters = list(layer.parameters())[2:]
            else:
                layer_parameters = list(layer.parameters())
            trainable_params += sum(p.numel() for p in layer_parameters if p.requires_grad)
            all_used_params += sum(p.numel() for p in layer_parameters)
        print('trainable_params:', trainable_params)
        print(' all_used_params:', all_used_params)
        print('               %:', round(trainable_params / all_used_params * 100, 2))

    def _calculate_common_params(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        common_params = sum(p.numel() for p in list(self.gpt_model.parameters())[2:])
        print('common_params:', common_params)
        print('   all_params:', all_params)
        print('            %:', round(common_params / all_params * 100, 2))
        print('trainable_params:', trainable_params)
        print('               %:', round(trainable_params / all_params * 100, 2))


class CrossAttentionGPT2FusionBrain(BaseGPT2FusionBrain):

    def __init__(self,
                 gpt_model,
                 attention_config,
                 handwritten_config,
                 vqa_config,
                 detection_config,
                 **freeze_gpt_kwargs):
        super().__init__(gpt_model, handwritten_config, **freeze_gpt_kwargs)

        ## zhOD[image, text] and VQA[image, text] layers:
        self.attention_config = attention_config
        self.backbone = ResnetBackbone(pretrained=True)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.embedding_size, kernel_size=1)
        self.text_projection = nn.Linear(self.embedding_size, self.embedding_size)
        self.image_projection = nn.Linear(self.embedding_size, self.embedding_size)
        self.cross_attention = nn.ModuleList([
            CrossAttentionLayer(
                self.embedding_size,
                attention_config['num_heads'],
                attention_config['pf_dim'],
                attention_config['dropout']
            )
            for _ in range(attention_config['num_attention_layers'])
        ])
        #####

        # detection[image, text] input/output layers:
        self.detection_config = detection_config
        self.detection_pool = nn.AdaptiveMaxPool2d((detection_config["num_queries"], None))
        self.bbox_embed = MLP(self.embedding_size, self.embedding_size, 5, detection_config['num_mlp_layers'])
        print('=== DETECTION TASK ===')
        self._calculate_trainable_params([
            self.backbone,
            self.gpt_model,
            self.input_proj,
            self.cross_attention,
            self.bbox_embed
        ])
        print('=== === === === ===')
        #####

        # vqa[image, text] input/output layers:
        self.vqa_config = vqa_config
        self.tokens_embed = nn.Linear(self.embedding_size, vqa_config['tokens_num'])
        print('=== VQA TASK ===')
        self._calculate_trainable_params([
            self.backbone,
            self.gpt_model,
            self.input_proj,
            self.cross_attention,
            self.bbox_embed
        ])
        print('=== === === === ===')
        #####

        print('=== COMMON PARAMS ===')
        self._calculate_common_params()
        print('=== === === === ===')

    def forward_vqa(self, images, tokens, labels):
        back_out = self.backbone(images)
        patchs = self.input_proj(back_out).flatten(-2).transpose(-1, -2)
        attention_mask = torch.tensor(labels.clone().detach() != 0, dtype=torch.uint8)
        attention_mask = attention_mask.to(labels.device)
        # Fusion Brain
        img_embs = self.gpt_model(inputs_embeds=patchs).last_hidden_state
        tokens_embs = self.gpt_model(input_ids=tokens, attention_mask=attention_mask).last_hidden_state
        #####
        img_embs = self.image_projection(img_embs)
        tokens_embs = self.text_projection(tokens_embs)

        norm_img_emb = F.normalize(img_embs.mean(-2), p=2, dim=-1)
        norm_tokens_emb = F.normalize(tokens_embs.mean(-2), p=2, dim=-1)

        for layer in self.cross_attention:
            tokens_embs, _ = layer(tokens_embs, img_embs)

        output_logits = self.tokens_embed(tokens_embs)
        out = {
            'pred_logits': output_logits,
            'proj_queries': norm_img_emb,
            'proj_tokens': norm_tokens_emb,
        }

        return out

    def forward_detection(self, images, tokens, attention_masks):
        back_out = self.backbone(images)
        patchs = self.input_proj(back_out).flatten(-2).transpose(-1, -2)
        # Fusion Brain
        img_embs = self.gpt_model(inputs_embeds=patchs).last_hidden_state
        tokens_embs = self.gpt_model(input_ids=tokens, attention_mask=attention_masks).last_hidden_state
        #####
        img_embs = self.image_projection(img_embs)
        tokens_embs = self.text_projection(tokens_embs)

        norm_img_emb = F.normalize(img_embs.mean(-2), p=2, dim=-1)
        norm_tokens_emb = F.normalize(tokens_embs.mean(-2), p=2, dim=-1)

        text_masks = attention_masks.type(torch.bool)
        for layer in self.cross_attention:
            img_embs, _ = layer(img_embs, tokens_embs, ~text_masks)
        img_embs = self.detection_pool(img_embs)

        output_logits = self.bbox_embed(img_embs).sigmoid()
        out = {
            'pred_logits': output_logits,
            'proj_queries': norm_img_emb,
            'proj_tokens': norm_tokens_emb,
        }

        return out


class InverseAttentionGPT2FusionBrain(BaseGPT2FusionBrain):

    def __init__(self,
                 gpt_model,
                 handwritten_config,
                 vqa_config,
                 detection_config,
                 **freeze_gpt_kwargs):
        super().__init__(gpt_model, handwritten_config, **freeze_gpt_kwargs)

        ## zhOD[image, text] and VQA[image, text] layers:
        self.backbone = ResnetBackbone(pretrained=True)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.embedding_size, kernel_size=1)
        #####

        # detection[image, text] input/output layers:
        self.detection_config = detection_config
        self.max_boxes = detection_config['max_boxes']
        self.bbox_embed = MLP(self.embedding_size, self.embedding_size, 5, detection_config['num_mlp_layers'])
        print('=== DETECTION TASK ===')
        self._calculate_trainable_params([
            self.backbone,
            self.gpt_model,
            self.input_proj,
            self.bbox_embed
        ])
        print('=== === === === ===')
        #####

        # vqa[image, text] input/output layers:
        self.vqa_config = vqa_config
        self.tokens_embed = nn.Linear(self.embedding_size, vqa_config['tokens_num'])
        print('=== VQA TASK ===')
        self._calculate_trainable_params([
            self.backbone,
            self.gpt_model,
            self.input_proj,
            self.bbox_embed
        ])
        print('=== === === === ===')
        #####

        print('=== COMMON PARAMS ===')
        self._calculate_common_params()
        print('=== === === === ===')

    def forward_vqa(self, images, tokens):
        back_out = self.backbone(images)
        img_embeddings = self.input_proj(back_out).flatten(-2).transpose(-1, -2)

        tokens_num = tokens.shape[-1]
        tokens_embeddings = self.gpt_model.wte(tokens) + self.gpt_model.wpe(
            torch.arange(tokens.shape[1], device=tokens.device))
        embedings = torch.cat((img_embeddings, tokens_embeddings), dim=1)
        # Fusion Brain
        gpt_out = self.gpt_model(inputs_embeds=embedings).last_hidden_state
        #####
        output_logits = self.tokens_embed(gpt_out[:, -tokens_num:])

        return output_logits

    def forward_detection(self, images, tokens):
        back_out = self.backbone(images)
        img_embeddings = self.input_proj(back_out).flatten(-2).transpose(-1, -2)
        tokens_embeddings = self.gpt_model.wte(tokens) + self.gpt_model.wpe(
            torch.arange(tokens.shape[1], device=tokens.device))
        embedings = torch.cat((tokens_embeddings, img_embeddings), dim=1)
        # Fusion Brain
        gpt_out = self.gpt_model(inputs_embeds=embedings).last_hidden_state
        #####

        output_logits = self.bbox_embed(gpt_out[:, -self.max_boxes:]).sigmoid()
        out = {
            'pred_logits': output_logits,
        }

        return out
