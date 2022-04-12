import torch


def inverse_detection_evaluation(model, images, input_ids, treshhold):
    back_out = model.backbone(images)
    img_embeddings = model.input_proj(back_out).flatten(-2).transpose(-1, -2)

    boxes = []
    for tokens in input_ids:
        tokens_embeddings = model.gpt_model.wte(tokens) + model.gpt_model.wpe(torch.arange(tokens.shape[1], device=tokens.device))
        box_embeddings = model.query_embed.weight.unsqueeze(0)
        embedings = torch.cat((tokens_embeddings, img_embeddings, box_embeddings), dim=1)
        gpt_out = model.gpt_model(inputs_embeds=embedings).last_hidden_state

        num_boxes = box_embeddings.shape[1]
        output_classes = model.class_embed(gpt_out[:, -num_boxes:]).squeeze(-1).sigmoid()
        output_boxes = model.bbox_embed(gpt_out[:, -num_boxes:]).sigmoid()
        output_boxes = output_boxes[output_classes > treshhold]
        boxes.append(output_boxes)

    return boxes
