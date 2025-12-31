import torch, tokenizers

def translate(model, tokenizer, src_text, device, max_len=20, sample=False, temperture=0.7):
    encoded_text = tokenizer.encode(src_text)

    src_ids = torch.tensor(encoded_text.ids, dtype=torch.long).reshape(1,-1).to(device)
    src_masks = torch.tensor(encoded_text.attention_mask, dtype=torch.long).reshape(1,-1).to(device)
    tgt_ids = torch.tensor([tokenizer.token_to_id("[BOS]")], dtype=torch.long).reshape(1,-1).to(device)

    eos_token_id = tokenizer.token_to_id("[EOS]")
    model.eval()

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(src_ids, src_masks, tgt_ids)[:, :, -1] # (1, vocab_size)

        if sample:
            scaled_logits = logits / temperture
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = logits.argmax(dim=1, keepdim=True)

        tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)

        if next_token_id.item() == eos_token_id:
            break

    return tokenizer.decode(tgt_ids.squeeze(0).cpu().numpy())
