import torch


def inverse_vqa_evaluation(model, images, tokens, max_answer_length):
    back_out = model.backbone(images)
    img_embeddings = model.input_proj(back_out).flatten(-2).transpose(-1, -2)

    answer_logits = []
    for _ in range(max_answer_length):
        tokens_embeddings = model.gpt_model.wte(tokens) + model.gpt_model.wpe(
            torch.arange(tokens.shape[1], device=tokens.device))
        embedings = torch.cat((img_embeddings, tokens_embeddings), dim=1)
        gpt_out = model.gpt_model(inputs_embeds=embedings).last_hidden_state

        logits = model.tokens_embed(gpt_out)
        last_logits = logits[:, -1, :]
        answer_logits.append(last_logits)

        new_tokens = torch.argmax(last_logits.softmax(-1), dim=-1, keepdim=True)
        tokens = torch.cat([tokens, new_tokens], dim=-1)

    return torch.stack(answer_logits, dim=1)
