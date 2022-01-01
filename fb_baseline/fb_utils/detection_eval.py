import torch
import torch.nn.functional as F


def cross_detection_evaluation(model, images, input_ids, attention_masks, cor_treshhold, treshhold):
    back_out = model.backbone(images)
    patchs = model.input_proj(back_out).flatten(-2).transpose(-1, -2)
    gpt_img = model.gpt_model(inputs_embeds=patchs).last_hidden_state
    norm_gpt_img = F.normalize(gpt_img, p=2, dim=-1)

    boxes = []
    for tokens, attention_mask in zip(input_ids, attention_masks):
        gpt_text = model.gpt_model(input_ids=tokens, attention_mask=attention_mask).last_hidden_state
        norm_gpt_text = F.normalize(gpt_text, p=2, dim=-1)
        corr_matrix = torch.matmul(norm_gpt_img, norm_gpt_text.transpose(-1, -2))
        cut_gpt_img = gpt_img[corr_matrix.mean(-1) > cor_treshhold].unsqueeze(0)
        if cut_gpt_img[1] == 0:
            boxes.append(torch.tensor([]).to(cut_gpt_img.device))
            continue
        text_mask = attention_mask.type(torch.bool)
        for layer in model.cross_attention:
            cut_gpt_img, _ = layer(cut_gpt_img, gpt_text, ~text_mask)
        cut_gpt_img = model.detection_pool(cut_gpt_img)

        output_logits = model.bbox_embed(cut_gpt_img).sigmoid()
        output_boxes = output_logits[output_logits[:, :, -1] > treshhold][:, :-1]
        boxes.append(output_boxes)

    return boxes


def inverse_detection_evaluation(model, images, input_ids, treshhold):
    back_out = model.backbone(images)
    img_embeddings = model.input_proj(back_out).flatten(-2).transpose(-1, -2)

    boxes = []
    for tokens in input_ids:
        tokens_embeddings = model.gpt_model.wte(tokens) + model.gpt_model.wpe(
            torch.arange(tokens.shape[1], device=tokens.device))
        embedings = torch.cat((tokens_embeddings, img_embeddings), dim=1)
        gpt_out = model.gpt_model(inputs_embeds=embedings).last_hidden_state

        output_logits = model.bbox_embed(gpt_out[:, -8:]).sigmoid()
        output_boxes = output_logits[output_logits[:, :, -1] > treshhold][:, :-1]
        boxes.append(output_boxes)

    return boxes
