**AttentionNMT🌐** is a English to Spanish Neural Machine Translation system implemented in PyTorch, built on a GRU‑based Encoder–Decoder architecture with Attention. Trained on 220k English–Spanish sentence pairs using a custom BPE tokenizer, it aligns source and target text at the subword level while overcoming the fixed‑length bottleneck of traditional recurrent models.

At the core of the design lies the **Attention Mechanism**, which allows the decoder to dynamically focus on different parts of the source sequence during translation. Instead of compressing an entire sentence into a single fixed‑size vector, attention provides a weighted context at each decoding step. This effectively extends the usable sequence length, enabling the model to handle longer sentences and complex grammatical dependencies with greater accuracy and fluency.

During inference, the model generates translations autoregressively: given a source sentence, it predicts tokens step by step until an [EOS] marker or a maximum length is reached. 

**✍️ Punctuation Tip** 
To improve translation quality, include proper punctuation in the english source text:  
- End **declarative sentences** with a period (`.`)  
- End **questions** with a question mark (`?`)  
- Use exclamation marks (`!`) where appropriate

Full information and source code for the project are available on [GitHUB](https://github.com/hoom4n/AttentionNMT).