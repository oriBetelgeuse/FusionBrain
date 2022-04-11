import torch


def inverse_detection_evaluation(model, images, input_ids, treshhold):
    back_out = model.backbone(images)
    img_embeddings = model.input_proj(back_out).flatten(-2).transpose(-1, -2)

    boxes = []
    for tokens in input_ids:
        tokens_embeddings = model.gpt_model.wte(tokens) + model.gpt_model.wpe(
            torch.arange(tokens.shape[1], device=tokens.device))
        embedings = torch.cat((tokens_embeddings, img_embeddings), dim=1)
        gpt_out = model.gpt_model(inputs_embeds=embedings).last_hidden_state

        output_logits = model.bbox_embed(gpt_out[:, -model.max_boxes:]).sigmoid()
        output_boxes = output_logits[output_logits[:, :, -1] > treshhold][:, :-1]
        boxes.append(output_boxes)

    return boxes
