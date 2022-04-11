import torch
from torch import nn
from einops import rearrange

from .modules.layers import ResnetBackbone, MLP


class InverseAttentionGPT2FusionBrain(nn.Module):

    def __init__(self, gpt_model, handwritten_config, vqa_config, detection_config, **freeze_gpt_kwargs):
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

        ## zhOD[image, text] and VQA[image, text] layers:
        self.backbone = ResnetBackbone(pretrained=True)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.embedding_size, kernel_size=1)
        #####

        # vqa[image, text] input/output layers:
        self.vqa_config = vqa_config
        self.tokens_embed = nn.Linear(self.embedding_size, vqa_config['tokens_num'])
        print('=== VQA TASK ===')
        self._calculate_trainable_params([
            self.backbone,
            self.gpt_model,
            self.input_proj,
            self.tokens_embed
        ])
        print('=== === === === ===')
        #####

        # detection[image, text] input/output layers:
        self.detection_config = detection_config
        self.query_embed = nn.Embedding(detection_config['num_queries'], self.embedding_size)
        self.class_embed = nn.Linear(self.embedding_size, 1)
        self.bbox_embed = MLP(self.embedding_size, self.embedding_size, 4, detection_config['num_mlp_layers'])
        print('=== DETECTION TASK ===')
        self._calculate_trainable_params([
            self.backbone,
            self.gpt_model,
            self.input_proj,
            self.query_embed,
            self.class_embed,
            self.bbox_embed
        ])
        print('=== === === === ===')
        #####

        self.forward_tasks = {
            'handwritten': self.forward_handwritten,
            'trans': self.forward_trans,
            'vqa': self.forward_vqa,
            'detection': self.forward_detection,
        }

        print('=== COMMON PARAMS ===')
        self._calculate_common_params()
        print('=== === === === ===')

    def forward(self, task_id, **kwargs):
        return self.forward_tasks[task_id](**kwargs)

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

    def forward_trans(self, input_ids, attention_mask):
        # Fusion Brain
        gpt_out = self.gpt_model(input_ids, attention_mask=attention_mask).last_hidden_state
        #####
        output_logits = self.lm_head(gpt_out)

        return output_logits

    def forward_vqa(self, images, tokens, attention_mask):
        back_out = self.backbone(images)
        img_embeddings = self.input_proj(back_out).flatten(-2).transpose(-1, -2)
        tokens_embeddings = self.gpt_model.wte(tokens) + self.gpt_model.wpe(torch.arange(tokens.shape[1], device=tokens.device))

        additiomal_attention_mask = torch.ones(
            attention_mask.shape[0], img_embeddings.shape[1], dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat((additiomal_attention_mask, attention_mask), dim=1)
        embedings = torch.cat((img_embeddings, tokens_embeddings), dim=1)
        # Fusion Brain
        gpt_out = self.gpt_model(inputs_embeds=embedings, attention_mask=attention_mask).last_hidden_state
        #####
        tokens_num = tokens.shape[1]
        output_logits = self.tokens_embed(gpt_out[:, -tokens_num:])

        return output_logits

    def forward_detection(self, images, tokens, attention_mask):
        bs = images.shape[0]

        back_out = self.backbone(images)
        img_embeddings = self.input_proj(back_out).flatten(-2).transpose(-1, -2)
        tokens_embeddings = self.gpt_model.wte(tokens) + self.gpt_model.wpe(torch.arange(tokens.shape[1], device=tokens.device))
        box_embeddings = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        additional_attention_mask = torch.ones(
            bs, img_embeddings.shape[1] + box_embeddings.shape[1], dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat((attention_mask, additional_attention_mask), dim=1)
        embedings = torch.cat((tokens_embeddings, img_embeddings, box_embeddings), dim=1)
        # Fusion Brain
        gpt_out = self.gpt_model(inputs_embeds=embedings, attention_mask=attention_mask).last_hidden_state
        #####

        num_boxes = box_embeddings.shape[1]
        output_classes = self.class_embed(gpt_out[:, -num_boxes:]).squeeze(-1).sigmoid()
        output_boxes = self.bbox_embed(gpt_out[:, -num_boxes:]).sigmoid()
        out = {
            'pred_classes': output_classes,
            'pred_boxes': output_boxes
        }

        return out

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
