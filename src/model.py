import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    """
    Scaled Dot Product Attention (Luong-Style) with Masking Support.

    Args:
        query: (N, Lq, Dq)
        key:   (N, Lk, Dk)
        value: (N, Lv, Dv), optional. If None, value=key
        mask:  (N, Lk), optional. 1 for valid tokens, 0 for padding.

    Returns:
        context: (N, Lq, Dv)
    """
    def __init__(self, use_scale=True):
        super().__init__()
        self.use_scale = use_scale

    def forward(self, query, key, value=None, mask=None):
        assert query.shape[-1] == key.shape[-1], "query & key must have same hidden dim"     
        if value is None:
            value = key
        else:
            assert key.shape[1] == value.shape[1], "key & value must have same sequence length"

        scale_factor = 1 / math.sqrt(key.size(-1)) if self.use_scale else 1 # 1 / sqer(Dk)
        attention_scores = query @ key.transpose(-2, -1) * scale_factor  # (N, Lq, Lk)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, Lk)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)  # (N, Lq, Lk)
        return attention_weights @ value  # (N, Lq, Dv)

class NMTModel(nn.Module):
    """
    Neural Machine Translation (NMT) model with GRU encoder–decoder and Luong-style attention.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int, optional): Dimension of token embeddings. Default: 512.
        hidden_dim (int, optional): Dimension of GRU hidden states. Default: 512.
        gru_layers (int, optional): Number of GRU layers for encoder and decoder. Default: 2.
        gru_dropout (float, optional): Dropout probability between GRU layers. Default: 0.1.
        pad_token_id (int, optional): Index of the padding token. Default: 0.

    Forward Inputs:
        input_ids (LongTensor): Source token IDs, shape (N, L_src).
        attention_mask (LongTensor): Source mask, shape (N, L_src), 1 for valid tokens.
        decoder_input_ids (LongTensor): Target token IDs, shape (N, L_tgt).
        decoder_attention_mask (LongTensor, optional): Target mask, shape (N, L_tgt).

    Forward Returns:
        logits (FloatTensor): Prediction scores, shape (N, vocab_size, L_tgt).
    """
    def __init__(self, vocab_size, embedding_dim = 512,
                 hidden_dim = 512, gru_layers = 2,
                 gru_dropout = 0.1, pad_token_id = 0):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.encoder = nn.GRU(embedding_dim, hidden_dim,
                              num_layers = gru_layers, batch_first=True,
                              dropout = gru_dropout)
        self.decoder = nn.GRU(embedding_dim, hidden_dim,
                              num_layers = gru_layers, batch_first=True,
                              dropout = gru_dropout)
        self.attention = Attention()
        self.output = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask=None):

        # source and target embedding
        src_embeddings = self.embedding(input_ids) # (N, L_src, emb_dim)
        tgt_embeddings = self.embedding(decoder_input_ids) # (N, L_tgt, emb_dim)

        # encoder
        src_lenghts = attention_mask.sum(dim=1)
        packed_encoder_inputs = pack_padded_sequence(src_embeddings,
                                                     src_lenghts.cpu(),
                                                     batch_first=True,
                                                     enforce_sorted=False)

        encoder_outputs_packed , encoder_hidden = self.encoder(packed_encoder_inputs) # encoder_hidden: (num_layers, N, H_enc)
        encoder_outputs, _ =  pad_packed_sequence(encoder_outputs_packed, batch_first=True) # (N, L_src, H_enc)

        # decoder
        decoder_outputs, _ = self.decoder(tgt_embeddings, encoder_hidden) # decoder_outputs: (N, L_tgt, H_dec)

        # attention
        attention_outputs = self.attention(query = decoder_outputs,
                                           key = encoder_outputs,
                                           mask = attention_mask) # (N, L_tgt, H_enc)

        # output
        cat_outputs = torch.concat([decoder_outputs, attention_outputs], dim=-1) # (N, L_tgt, H_dec + H_enc) 
        logits = self.output(cat_outputs) # (N, L_tgt, vocab_size)
        return logits.permute(0, 2, 1) # (N, vocab_size, L)
