#!/usr/bin/env python3
"""Builds 03_train_student.ipynb as valid JSON."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "03_train_student.ipynb"

def md(cid, src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type":"markdown","id":cid,"metadata":{},"source":source}

def code(cid, src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type":"code","execution_count":None,"id":cid,"metadata":{},"outputs":[],"source":source}

cells = []

# ── MARKDOWN: Title ───────────────────────────────────────────────────────
cells.append(md("md00", """\
# Notebook 03 — Student Model Training
## Multi-Hypothesis Distillation  |  English → Swahili

**Paper:** *Multi-Hypothesis Distillation of Multilingual Neural Translation Models for Low-Resource Languages*

### Skip-if-done logic
Every dataset variant checks for an existing `best_checkpoint.pt` **and** a
`pred_<key>_devtest.txt` before training. If both exist, that variant is skipped
automatically — re-run the notebook any time without duplicating work.

### Expected folder layout
```
MHD2/
  data/synthetic/   eng_swh_{beam_M1,beam_M10,dbs_M10,mbr_M10,top_k_M10,top_p_M10}.jsonl
  data/flores/      dev.jsonl  devtest.jsonl
  models/           spm_*/  student_*/   (written here)
  results/predictions/       (written here)
  results/translation/       (written here)
```"""))

# ── CELL 01: Install ──────────────────────────────────────────────────────
cells.append(code("cell01", """\
# Cell 1 — Install dependencies (safe to re-run)
import subprocess, sys

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("sentencepiece==0.2.0", "sacrebleu==2.4.2", "tqdm")
print("Dependencies ready.")"""))

# ── CELL 02: Imports ──────────────────────────────────────────────────────
cells.append(code("cell02", """\
# Cell 2 — Imports
import os, json, math, time, random, csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AMP — compatible with PyTorch 1.x and 2.x
try:
    from torch.amp import GradScaler, autocast   # PyTorch >= 2.0
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch 1.x
    _AMP_NEW = False

import sentencepiece as spm
from sacrebleu.metrics import BLEU, CHRF
from tqdm.auto import tqdm

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")"""))

# ── CELL 03: Paths ────────────────────────────────────────────────────────
cells.append(code("cell03", """\
# Cell 3 — Path configuration
# =========================================================
# KAGGLE:       uncomment the two lines below
# WORKSPACE = Path("/kaggle/input/your-dataset-name")  # READ-ONLY
# OUT_ROOT  = Path("/kaggle/working")                  # writable
#
# LIGHTNING AI: uncomment the two lines below
# WORKSPACE = Path("/teamspace/studios/this_studio/MHD2")
# OUT_ROOT  = Path("/teamspace/studios/this_studio/MHD2")
#
# LOCAL (default):
WORKSPACE = Path(r"C:/Users/nirmi/Desktop/MHD2")
OUT_ROOT  = WORKSPACE
# =========================================================

DATA_SYNTHETIC = WORKSPACE / "data" / "synthetic"
DATA_FLORES    = WORKSPACE / "data" / "flores"
MODELS_DIR     = OUT_ROOT  / "models"
RESULTS_DIR    = OUT_ROOT  / "results"
PRED_DIR       = RESULTS_DIR / "predictions"
TRANS_DIR      = RESULTS_DIR / "translation"

for d in [MODELS_DIR, PRED_DIR, TRANS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SYNTHETIC_FILES = {
    "beam_M1"  : DATA_SYNTHETIC / "eng_swh_beam_M1.jsonl",
    "beam_M10" : DATA_SYNTHETIC / "eng_swh_beam_M10.jsonl",
    "dbs_M10"  : DATA_SYNTHETIC / "eng_swh_dbs_M10.jsonl",
    "mbr_M10"  : DATA_SYNTHETIC / "eng_swh_mbr_M10.jsonl",
    "top_k_M10": DATA_SYNTHETIC / "eng_swh_top_k_M10.jsonl",
    "top_p_M10": DATA_SYNTHETIC / "eng_swh_top_p_M10.jsonl",
}

FLORES_DEV     = DATA_FLORES / "dev.jsonl"
FLORES_DEVTEST = DATA_FLORES / "devtest.jsonl"

print("=== Path check ===")
all_ok = True
for name, p in {**SYNTHETIC_FILES,
                "flores_dev": FLORES_DEV,
                "flores_devtest": FLORES_DEVTEST}.items():
    ok = p.exists()
    if not ok:
        all_ok = False
    print(f"  [{'OK' if ok else 'MISSING'}] {name}")
if not all_ok:
    print("\\nWARNING: some files missing — fix paths above before continuing.")
else:
    print("\\nAll data files found.")"""))

# ── CELL 04: Hyperparameters ──────────────────────────────────────────────
cells.append(code("cell04", """\
# Cell 4 — Hyperparameters
# Change MODEL_OPTION and TRAIN_DATASET_KEY here to train a different variant.

MODEL_OPTION      = "A"        # "A"=65M paper model | "B"=tiny debug
TRAIN_DATASET_KEY = "beam_M1"  # key from SYNTHETIC_FILES

# Tokenizer
SPM_VOCAB_SIZE = 10_000
SPM_MODEL_TYPE = "bpe"

# Architecture  (Option A = paper-faithful ~65M)
CFG_A = dict(encoder_layers=6, decoder_layers=6,
             d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
# Architecture  (Option B = tiny debug ~7M)
CFG_B = dict(encoder_layers=2, decoder_layers=2,
             d_model=128, nhead=4, dim_feedforward=512, dropout=0.1)

MODEL_CFG = CFG_A if MODEL_OPTION == "A" else CFG_B

# Training
MAX_LEN             = 150
BATCH_SIZE          = 32
GRAD_ACCUM_STEPS    = 2
LEARNING_RATE       = 7e-4
ADAM_BETAS          = (0.9, 0.98)
ADAM_EPS            = 1e-9
WARMUP_STEPS        = 8000
LABEL_SMOOTHING     = 0.1
NUM_EPOCHS          = 50
EARLY_STOP_PATIENCE = 6
BEAM_SIZE           = 5
USE_AMP             = True

# Set False to skip per-epoch FLORES beam-eval (speeds up large datasets)
EVAL_DURING_TRAINING = True

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_amp_enabled = USE_AMP and DEVICE.type == "cuda"

print(f"Device : {DEVICE}")
print(f"Model  : Option {MODEL_OPTION}  ({MODEL_CFG})")
print(f"Dataset: {TRAIN_DATASET_KEY}")
print(f"AMP    : {_amp_enabled}")"""))

# ── CELL 05: Data loading ─────────────────────────────────────────────────
cells.append(code("cell05", """\
# Cell 5 — Load data

def load_synthetic(path: Path) -> List[Dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                pairs.append({"src": obj["src"].strip(),
                               "tgt": obj["tgt"].strip()})
    return pairs

def load_flores(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                records.append({"id": obj["id"],
                                 "src": obj["source"].strip(),
                                 "reference": obj["reference"].strip()})
    return records

train_pairs    = load_synthetic(SYNTHETIC_FILES[TRAIN_DATASET_KEY])
flores_dev     = load_flores(FLORES_DEV)
flores_devtest = load_flores(FLORES_DEVTEST)

print(f"Train pairs    : {len(train_pairs):,}")
print(f"FLORES dev     : {len(flores_dev):,}")
print(f"FLORES devtest : {len(flores_devtest):,}")
print()
print("Sample train  SRC:", train_pairs[0]["src"][:80])
print("Sample train  TGT:", train_pairs[0]["tgt"][:80])
print("Sample dev    SRC:", flores_dev[0]["src"][:80])
print("Sample dev    REF:", flores_dev[0]["reference"][:80])"""))

# ── CELL 06: Tokenizer ────────────────────────────────────────────────────
cells.append(code("cell06", """\
# Cell 6 — SentencePiece tokenizer (skip if already trained)

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

SPM_DIR          = MODELS_DIR / f"spm_eng_swh_{TRAIN_DATASET_KEY}"
SPM_DIR.mkdir(parents=True, exist_ok=True)
SPM_MODEL_PREFIX = str(SPM_DIR / "spm")
SPM_MODEL_PATH   = SPM_DIR / "spm.model"

if SPM_MODEL_PATH.exists():
    print(f"[SKIP] SPM model already exists: {SPM_MODEL_PATH}")
else:
    print("Training SentencePiece tokenizer...")
    tmp_txt = SPM_DIR / "spm_train_text.txt"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(pair["src"].replace("\\n", " ") + "\\n")
            f.write(pair["tgt"].replace("\\n", " ") + "\\n")
        for r in flores_dev + flores_devtest:
            f.write(r["src"].replace("\\n", " ") + "\\n")
            f.write(r["reference"].replace("\\n", " ") + "\\n")
    spm.SentencePieceTrainer.train(
        input=str(tmp_txt),
        model_prefix=SPM_MODEL_PREFIX,
        vocab_size=SPM_VOCAB_SIZE,
        model_type=SPM_MODEL_TYPE,
        character_coverage=0.9995,
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        user_defined_symbols=[],
        max_sentence_length=4096,
        shuffle_input_sentence=True,
        input_sentence_size=5_000_000,
    )
    tmp_txt.unlink(missing_ok=True)
    print(f"SPM saved: {SPM_MODEL_PATH}")

sp = spm.SentencePieceProcessor()
sp.load(str(SPM_MODEL_PATH))
VOCAB_SIZE = sp.get_piece_size()
print(f"Vocab size: {VOCAB_SIZE}")

# Round-trip check
_t = "The quick brown fox."
assert sp.decode(sp.encode(_t)) == _t, "Round-trip FAILED"
print("Round-trip OK:", repr(_t))"""))

# ── CELL 07: Dataset + DataLoader ─────────────────────────────────────────
cells.append(code("cell07", """\
# Cell 7 — Dataset and DataLoader

class TranslationDataset(Dataset):
    def __init__(self, pairs, sp_model, max_len=150):
        self.pairs   = pairs
        self.sp      = sp_model
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def _enc(self, text):
        ids = [BOS_ID] + self.sp.encode(text) + [EOS_ID]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len - 1] + [EOS_ID]
        return ids

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return (torch.tensor(self._enc(p["src"]), dtype=torch.long),
                torch.tensor(self._enc(p["tgt"]), dtype=torch.long))


class FLORESDataset(Dataset):
    def __init__(self, records, sp_model, max_len=150):
        self.records = records
        self.sp      = sp_model
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ids = [BOS_ID] + self.sp.encode(self.records[idx]["src"]) + [EOS_ID]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len - 1] + [EOS_ID]
        return torch.tensor(ids, dtype=torch.long)


def collate_pairs(batch):
    src_b, tgt_b = zip(*batch)
    src_p = torch.nn.utils.rnn.pad_sequence(src_b, batch_first=True, padding_value=PAD_ID)
    tgt_p = torch.nn.utils.rnn.pad_sequence(tgt_b, batch_first=True, padding_value=PAD_ID)
    return src_p, tgt_p, torch.tensor([len(s) for s in src_b]), torch.tensor([len(t) for t in tgt_b])


def collate_src(batch):
    padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    return padded, torch.tensor([len(s) for s in batch])


val_size     = max(200, int(0.05 * len(train_pairs)))
train_subset = train_pairs[:len(train_pairs) - val_size]
val_subset   = train_pairs[len(train_pairs) - val_size:]

_pin = (DEVICE.type == "cuda")
train_ds = TranslationDataset(train_subset, sp, MAX_LEN)
val_ds   = TranslationDataset(val_subset,   sp, MAX_LEN)
dev_ds   = FLORESDataset(flores_dev,     sp, MAX_LEN)
dts_ds   = FLORESDataset(flores_devtest, sp, MAX_LEN)

train_loader   = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_pairs, num_workers=0, pin_memory=_pin)
val_loader     = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_pairs, num_workers=0, pin_memory=_pin)
dev_loader     = DataLoader(dev_ds,   32,         shuffle=False, collate_fn=collate_src,   num_workers=0)
devtest_loader = DataLoader(dts_ds,   32,         shuffle=False, collate_fn=collate_src,   num_workers=0)

print(f"Train batches : {len(train_loader)} ({len(train_ds):,} sentences)")
print(f"Val   batches : {len(val_loader)}  ({len(val_ds):,} sentences)")"""))

# ── CELL 08: Model ────────────────────────────────────────────────────────
cells.append(code("cell08", """\
# Cell 8 — Student Transformer model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, encoder_layers=6,
                 decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_len=512, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos     = PositionalEncoding(d_model, dropout, max_len + 10)
        self.tr      = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False)
        self.proj    = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = self.embed.weight  # weight tying
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _pad_mask(self, x):
        return x == self.pad_idx

    def _causal_mask(self, sz, device):
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device)

    def encode(self, src):
        kpm  = self._pad_mask(src)
        emb  = self.pos(self.embed(src) * math.sqrt(self.d_model))
        mem  = self.tr.encoder(emb, src_key_padding_mask=kpm)
        return mem, kpm

    def decode(self, tgt, mem, mem_kpm=None):
        t    = tgt.size(1)
        tmsk = self._causal_mask(t, tgt.device)
        tkpm = self._pad_mask(tgt)
        emb  = self.pos(self.embed(tgt) * math.sqrt(self.d_model))
        out  = self.tr.decoder(emb, mem, tgt_mask=tmsk,
                               tgt_key_padding_mask=tkpm,
                               memory_key_padding_mask=mem_kpm)
        return self.proj(out)

    def forward(self, src, tgt):
        mem, kpm = self.encode(src)
        return self.decode(tgt, mem, kpm)


model = StudentTransformer(
    vocab_size=VOCAB_SIZE, max_len=MAX_LEN, pad_idx=PAD_ID, **MODEL_CFG
).to(DEVICE)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}  (~{total/1e6:.1f}M)  |  Option {MODEL_OPTION}")"""))

# ── CELL 09: Loss / Optimizer / Scheduler ─────────────────────────────────
cells.append(code("cell09", """\
# Cell 9 — Loss, optimizer, scheduler

class LabelSmoothedCE(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.V   = vocab_size
        self.pad = pad_idx
        self.eps = smoothing
        self.cf  = 1.0 - smoothing

    def forward(self, logits, target):
        with torch.no_grad():
            sv = self.eps / (self.V - 2)
            st = torch.full_like(logits, sv)
            st.scatter_(1, target.unsqueeze(1), self.cf)
            st[:, self.pad] = 0.0
        pad_mask = target == self.pad
        lp   = F.log_softmax(logits, dim=-1)
        loss = -(st * lp).sum(-1).masked_fill(pad_mask, 0.0)
        return loss.sum() / (~pad_mask).sum().float().clamp(min=1)


criterion = LabelSmoothedCE(VOCAB_SIZE, PAD_ID, LABEL_SMOOTHING).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS)

def lr_lambda(step):
    step = max(step, 1)
    d    = MODEL_CFG["d_model"]
    return (d ** -0.5) * min(step ** -0.5, step * WARMUP_STEPS ** -1.5) / LEARNING_RATE

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if _AMP_NEW:
    scaler = GradScaler(device="cuda", enabled=_amp_enabled)
else:
    scaler = GradScaler(enabled=_amp_enabled)

print("Criterion / optimizer / scheduler ready.")
print(f"  AMP: {_amp_enabled}  |  Warmup steps: {WARMUP_STEPS}")"""))

# ── CELL 10: Beam search + evaluation ────────────────────────────────────
cells.append(code("cell10", """\
# Cell 10 — Beam search and evaluation helpers

chrf_metric = CHRF(word_order=2)   # chrF++
bleu_metric = BLEU(tokenize="13a")


@torch.no_grad()
def beam_search(model, src, beam_size=5, max_len=150):
    model.eval()
    device = src.device
    mem, kpm = model.encode(src)
    beams     = [(0.0, [BOS_ID])]
    completed = []

    for _ in range(max_len):
        cands = []
        for lp, ids in beams:
            if ids[-1] == EOS_ID:
                completed.append((lp, ids)); continue
            tgt_t   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits  = model.decode(tgt_t, mem, kpm)
            lps     = F.log_softmax(logits[0, -1], dim=-1)
            top_lp, top_id = lps.topk(beam_size)
            for l, i in zip(top_lp.tolist(), top_id.tolist()):
                cands.append((lp + l, ids + [i]))

        def scored(item):
            lp, seq = item
            return lp / (((5 + len(seq)) / 6) ** 1.0)

        cands.sort(key=scored, reverse=True)
        beams = []
        for lp, seq in cands:
            if seq[-1] == EOS_ID:
                completed.append((lp, seq))
            else:
                beams.append((lp, seq))
            if len(beams) >= beam_size:
                break
        if not beams:
            break

    pool = completed if completed else beams
    pool.sort(key=lambda x: x[0] / (((5 + len(x[1])) / 6) ** 1.0), reverse=True)
    best = pool[0][1]
    if best and best[0]  == BOS_ID: best = best[1:]
    if best and best[-1] == EOS_ID: best = best[:-1]
    return best


@torch.no_grad()
def translate_loader(model, loader, sp_model, beam_size=5, verbose=True):
    model.eval()
    hyps = []
    it   = tqdm(loader, desc="Translating") if verbose else loader
    for src_pad, src_lens in it:
        src_pad = src_pad.to(DEVICE)
        for i in range(src_pad.size(0)):
            src_i = src_pad[i, :src_lens[i]].unsqueeze(0)
            ids   = beam_search(model, src_i, beam_size, MAX_LEN)
            hyps.append(sp_model.decode(ids))
    return hyps


def score_translations(hyps, refs, prefix=""):
    assert len(hyps) == len(refs)
    chrf = chrf_metric.corpus_score(hyps, [refs]).score
    bleu = bleu_metric.corpus_score(hyps, [refs]).score
    print(f"{prefix}chrF++ = {chrf:.2f}  |  BLEU = {bleu:.2f}")
    return {"chrf++": round(chrf, 2), "bleu": round(bleu, 2)}


def append_scores_csv(key, epoch, split, scores):
    p = TRANS_DIR / f"scores_{key}.csv"
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","split","dataset","model","chrf++","bleu"])
        if new: w.writeheader()
        w.writerow({"epoch": epoch, "split": split, "dataset": key,
                    "model": MODEL_OPTION, **scores})


print("Beam search and evaluation helpers ready.")"""))

# ── CELL 11: Checkpointing ─────────────────────────────────────────────────
cells.append(code("cell11", """\
# Cell 11 — Checkpoint helpers

CKPT_DIR  = MODELS_DIR / f"student_eng_swh_{TRAIN_DATASET_KEY}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = CKPT_DIR / "best_checkpoint.pt"
LAST_CKPT = CKPT_DIR / "last_checkpoint.pt"


def save_ckpt(path, epoch, val_loss, extra=None):
    state = dict(epoch=epoch, val_loss=val_loss,
                 model_state=model.state_dict(),
                 optimizer_state=optimizer.state_dict(),
                 scheduler_state=scheduler.state_dict(),
                 model_cfg=MODEL_CFG, vocab_size=VOCAB_SIZE,
                 dataset_key=TRAIN_DATASET_KEY,
                 spm_path=str(SPM_MODEL_PATH))
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_ckpt(path, load_optimizer=True):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state"])
    if load_optimizer:
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
    print(f"  Loaded: {path}  (epoch={state['epoch']}, val_loss={state['val_loss']:.4f})")
    return state["epoch"], state["val_loss"]


print(f"Checkpoint dir: {CKPT_DIR}")"""))

# ── CELL 12: Sanity checks ────────────────────────────────────────────────
cells.append(code("cell12", """\
# Cell 12 — Sanity checks (run once before training)

print("=" * 55)
print("CHECK 1: Decode 5 untrained samples (expect garbage)")
print("=" * 55)
model.eval()
for i in range(5):
    src_t = torch.tensor([BOS_ID] + sp.encode(train_pairs[i]["src"]) + [EOS_ID],
                         dtype=torch.long, device=DEVICE).unsqueeze(0)
    ids   = beam_search(model, src_t, beam_size=3, max_len=40)
    print(f"  [{i}] SRC : {train_pairs[i]['src'][:60]}")
    print(f"  [{i}] PRED: {sp.decode(ids)[:60]}")
    print()

print("=" * 55)
print("CHECK 2: Overfit 100 examples (loss should drop)")
print("=" * 55)
_ods  = TranslationDataset(train_pairs[:100], sp, MAX_LEN)
_odl  = DataLoader(_ods, 16, shuffle=True, collate_fn=collate_pairs)
_m2   = StudentTransformer(VOCAB_SIZE, max_len=MAX_LEN, pad_idx=PAD_ID,
                           **CFG_B).to(DEVICE)  # always use tiny for speed
_opt2 = torch.optim.Adam(_m2.parameters(), lr=1e-3)
_cr2  = LabelSmoothedCE(VOCAB_SIZE, PAD_ID, 0.1).to(DEVICE)
_m2.train()
for step in range(50):
    for sb, tb, _, _ in _odl:
        sb, tb = sb.to(DEVICE), tb.to(DEVICE)
        loss = _cr2(_m2(sb, tb[:,:-1]).reshape(-1,VOCAB_SIZE), tb[:,1:].reshape(-1))
        _opt2.zero_grad(); loss.backward(); _opt2.step()
    if (step+1) % 10 == 0:
        print(f"  Step {step+1:3d} | loss = {loss.item():.4f}")
del _m2, _opt2, _cr2, _ods, _odl
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("  Expect loss < 2.0 by step 50. If not, check learning rate.")

print()
print("=" * 55)
print("CHECK 3: FLORES format")
print("=" * 55)
assert flores_dev[0]["src"] and flores_dev[0]["reference"]
print(f"  dev[0] id : {flores_dev[0]['id']}")
print(f"  dev[0] src: {flores_dev[0]['src'][:60]}")
print(f"  dev[0] ref: {flores_dev[0]['reference'][:60]}")
print("  FLORES format OK")"""))

# ── CELL 13: Training functions ───────────────────────────────────────────
cells.append(code("cell13", """\
# Cell 13 — Training and validation functions

def train_epoch(epoch, global_step):
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
    for bi, (sb, tb, _, _) in enumerate(pbar):
        sb, tb = sb.to(DEVICE, non_blocking=True), tb.to(DEVICE, non_blocking=True)
        with autocast(device_type=DEVICE.type, enabled=_amp_enabled):
            logits = model(sb, tb[:, :-1])
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tb[:, 1:].reshape(-1))
            loss   = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        if (bi + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        n += 1
        pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM_STEPS:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(n, 1), global_step


def val_epoch(epoch):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for sb, tb, _, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
            sb, tb = sb.to(DEVICE), tb.to(DEVICE)
            with autocast(device_type=DEVICE.type, enabled=_amp_enabled):
                logits = model(sb, tb[:, :-1])
                loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tb[:, 1:].reshape(-1))
            total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


print("train_epoch / val_epoch defined.")"""))

# ── CELL 14: Main training loop ───────────────────────────────────────────
cells.append(code("cell14", """\
# Cell 14 — Main training loop
# ─────────────────────────────────────────────────────────
# SKIP-IF-DONE: if best_checkpoint.pt already exists this
# variant is fully skipped.  Delete the file to retrain.
# ─────────────────────────────────────────────────────────

if BEST_CKPT.exists():
    print(f"[SKIP] Checkpoint already exists: {BEST_CKPT}")
    print("  Loading best model for evaluation ...")
    _ep, _vl = load_ckpt(BEST_CKPT, load_optimizer=False)
    print(f"  Epoch={_ep}  val_loss={_vl:.4f}")
    print("  To retrain, delete the checkpoint file and re-run this cell.")
else:
    best_val_loss    = float("inf")
    patience_counter = 0
    global_step      = 0
    history          = []
    start_epoch      = 1

    # Resume from last checkpoint if it exists (session restart recovery)
    if LAST_CKPT.exists():
        print(f"Resuming from {LAST_CKPT} ...")
        start_epoch, best_val_loss = load_ckpt(LAST_CKPT)
        start_epoch += 1

    print(f"Training {TRAIN_DATASET_KEY}  |  {len(train_ds):,} sentences")
    print(f"Epochs: {start_epoch} to {NUM_EPOCHS}  |  Patience: {EARLY_STOP_PATIENCE}")
    t0 = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        ep_t0 = time.time()

        train_loss, global_step = train_epoch(epoch, global_step)
        val_loss                = val_epoch(epoch)

        # Per-epoch FLORES dev evaluation
        if EVAL_DURING_TRAINING:
            dev_hyps   = translate_loader(model, dev_loader, sp,
                                          BEAM_SIZE, verbose=False)
            dev_refs   = [r["reference"] for r in flores_dev]
            dev_scores = score_translations(dev_hyps, dev_refs,
                                            prefix=f"  Ep {epoch:3d} | dev  ")
        else:
            dev_scores = {"chrf++": 0.0, "bleu": 0.0}

        # Always save last checkpoint (session-resume safety)
        save_ckpt(LAST_CKPT, epoch, val_loss)

        ep_time = time.time() - ep_t0
        lr_now  = scheduler.get_last_lr()[0]
        row = dict(epoch=epoch, train_loss=round(train_loss,4),
                   val_loss=round(val_loss,4),
                   dev_chrf=dev_scores["chrf++"],
                   dev_bleu=dev_scores["bleu"],
                   lr=round(lr_now,8), time_s=round(ep_time,1),
                   step=global_step)
        history.append(row)

        print(f"  Ep {epoch:3d}/{NUM_EPOCHS} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  {ep_time:.0f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_ckpt(BEST_CKPT, epoch, val_loss, extra=dev_scores)
            print(f"  ✓ Best val_loss={best_val_loss:.4f} — saved.")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\\n  Early stopping at epoch {epoch}.")
                break

    total_time = time.time() - t0
    print(f"\\nDone. Total: {total_time/60:.1f} min  |  Best val_loss: {best_val_loss:.4f}")"""))
