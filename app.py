import torch, tokenizers
import gradio as gr
from src.model import NMTModel
from src.config import HPARAMS
from src.inference import translate
from src.ui import build_demo

### CONFIGURATION ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch Device: {device}")
hp = HPARAMS()

### LOAD TOKENIZER ###
tokenizer = tokenizers.Tokenizer.from_file("tokenizer/bpe_tokenizer.json")

### LOAD MODEL ###
model = NMTModel(tokenizer.get_vocab_size(), **hp.model_hparams).to(device)
state_dict = torch.load("model/nmt_model_params.pt", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

### GRADIO APP ###
def translate_fn(src_text, max_len):
    return translate(model, tokenizer, src_text, device, max_len=max_len)

inputs = [
    gr.Textbox(label="English Text", lines=3),
    gr.Slider(10, 100, value=32, step=5, label="Max Length"),
]

outputs = [gr.Textbox(label="Spanish Translation", lines=5, interactive=False)]

demo =  build_demo(
    translate_fn,
    inputs,
    outputs,
    english_title = "# 🌐 AttentionNMT: GRU-Attention Encoder-Decoder NMT",
    persian_title = "# 🌐 ترجمه‌ی ماشینی انگلیسی به اسپانیایی با معماری رمزگذار–رمزگشا و Attention",
    assets_dir = "assets",
    app_title = "AttentionNMT"
)

demo.launch()
