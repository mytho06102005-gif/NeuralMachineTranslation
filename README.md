# NeuralMachineTranslation
- Mô hình sau khi train xong được up lên Hugging face: https://huggingface.co/Mytho0610/NeuralMachineTranslation
- Link tải dataset: https://huggingface.co/datasets/thainq107/iwslt2015-en-vi

Sau khi huấn luyện, mô hình được lưu trữ trên nền tảng Hugging Face nhằm hỗ trợ tái sử dụng và triển khai mà không cần thực hiện huấn luyện lại. Người dùng có thể tải trực tiếp mô hình đã huấn luyện để dịch văn bản mới.

Bước 1:
''' !pip install -q tokenizers huggingface_hub gradio
print("✅ Done") 
'''

Bước 2:
''' import torch
import torch.nn as nn
import math
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import gradio as gr

# ── Kiến trúc khớp đúng với checkpoint ────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # tên "pe.pe" khớp checkpoint

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerNMT(nn.Module):
    def __init__(self, vocab=16000, d_model=256, nhead=8,
                 num_enc=3, num_dec=3, dim_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, dropout=dropout)  # "pe.pe" ✅

        # "tf.encoder" / "tf.decoder" ✅
        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.proj = nn.Linear(d_model, vocab)  # "proj" ✅

    def forward(self, src, tgt, src_pad=None, tgt_pad=None):
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src.device)
        src_e = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_e = self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        out = self.tf(src_e, tgt_e,
                      tgt_mask=tgt_mask,
                      src_key_padding_mask=src_pad,
                      tgt_key_padding_mask=tgt_pad,
                      memory_key_padding_mask=src_pad)
        return self.proj(out)

    def encode(self, src, src_pad=None):
        src_e = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
        return self.tf.encoder(src_e, src_key_padding_mask=src_pad)

    def decode(self, tgt, mem, tgt_mask=None, tgt_pad=None, mem_pad=None):
        tgt_e = self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        return self.tf.decoder(tgt_e, mem,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_pad,
                               memory_key_padding_mask=mem_pad)

# ── Load từ HuggingFace ────────────────────────────────
REPO   = "Mytho0610/NeuralMachineTranslation"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⏳ Loading model từ {REPO} ...")

model_path = hf_hub_download(REPO, "best_model.pt")
tok_path   = hf_hub_download(REPO, "bpe_tokenizer.json")

model = TransformerNMT()
ckpt  = torch.load(model_path, map_location=device)
state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
model.load_state_dict(state)
model.to(device).eval()
print("✅ Model load thành công!")

tokenizer = Tokenizer.from_file(tok_path)
PAD, BOS, EOS = 0, 2, 3

# ── Hàm dịch Beam Search ──────────────────────────────
def translate(text, beam_size=4, max_len=128):
    if not text.strip():
        return ""
    ids  = tokenizer.encode(text).ids
    src  = torch.tensor([ids], device=device)
    smask = (src == PAD)

    with torch.no_grad():
        mem = model.encode(src, smask)

    beams, done = [(0.0, [BOS])], []

    for _ in range(max_len):
        cands = []
        for score, seq in beams:
            if seq[-1] == EOS:
                done.append((score / len(seq), seq))
                continue
            tgt  = torch.tensor([seq], device=device)
            tmask = nn.Transformer.generate_square_subsequent_mask(len(seq), device=device)
            with torch.no_grad():
                out   = model.decode(tgt, mem, tmask, None, smask)
                probs = torch.log_softmax(model.proj(out[:, -1]), dim=-1)
            vals, idxs = torch.topk(probs, beam_size, dim=-1)
            for v, i in zip(vals[0], idxs[0]):
                cands.append((score + v.item(), seq + [i.item()]))

        if not cands:
            break
        beams = sorted(cands, key=lambda x: x[0] / len(x[1]), reverse=True)[:beam_size]
        if all(s[-1] == EOS for _, s in beams):
            done += [(sc / len(s), s) for sc, s in beams]
            break

    best = sorted(done or beams, key=lambda x: x[0], reverse=True)[0][1]
    best = [t for t in best if t not in (BOS, EOS, PAD)]
    return tokenizer.decode(best)

with gr.Blocks(title="EN→VI Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌐 Dịch Anh → Việt\n*Model: Mytho0610/NeuralMachineTranslation*")
    with gr.Row():
        inp = gr.Textbox(label="🇬🇧 Tiếng Anh", placeholder="Nhập câu tiếng Anh...", lines=5)
        out = gr.Textbox(label="🇻🇳 Tiếng Việt", lines=5, interactive=False)
    with gr.Row():
        btn_translate = gr.Button("Dịch 🔁", variant="primary")
        btn_clear     = gr.Button("Xóa 🗑️", variant="secondary")
    
    btn_translate.click(fn=translate, inputs=inp, outputs=out)
    btn_clear.click(fn=lambda: ("", ""), inputs=None, outputs=[inp, out])
    
    gr.Examples(
        examples=[
            ["Hello, how are you?"],
            ["I love learning machine translation."],
            ["The weather is very nice today."],
            ["Can you help me with this problem?"],
        ],
        inputs=inp
    )

demo.launch(share=True)  '''
