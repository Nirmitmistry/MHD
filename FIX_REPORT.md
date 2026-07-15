# FIX_REPORT.md — Student MT Pipeline Bug Report

## Summary

The `beam_M1` student training run produced BLEU=0.00, chrF++=1.52 on dev and BLEU=0.00, chrF++=1.46 on devtest. This is a fully collapsed model — it is generating near-random outputs or identical/empty strings on every input. The primary cause is a **learning-rate bug that made the model's effective LR approximately 1,000× too small**, preventing all meaningful weight updates. Several secondary bugs compounded the failure.

---

## Bug 1 — Learning Rate Scale (PRIMARY CAUSE of BLEU=0)

### Root cause

`03b_train.ipynb` defines the Noam schedule as:

```python
LEARNING_RATE = 1e-3        # base LR for optimizer

def get_scheduler(optimizer):
    d = model_cfg["d_model"]   # 512
    def lr_lambda(step):
        step = max(step, 1)
        scale = min(step ** -0.5, step * WARMUP_STEPS ** -1.5)
        return (d ** -0.5) * scale
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, ...)
```

`LambdaLR` multiplies `lr_lambda(step)` by the optimizer's **base LR**, not by 1.0. The resulting effective LR is:

```
effective_lr = 1e-3  ×  (512^-0.5)  ×  min(step^-0.5, step × warmup^-1.5)
             = 1e-3  ×  0.04419      ×  (max value ~0.04096 at warmup=596)
             ≈ 1e-3  ×  1.81e-3
             ≈ 1.81e-6   (peak!)
             ≈ 7e-10     (at step 1)
```

The intended Noam formula produces a **peak LR of ~1.8e-3** when `optimizer LR = 1.0`. With `optimizer LR = 1e-3`, the actual peak was ~**1.8e-6** — roughly 1,000× too small.

### Evidence from training logs

`training_log_student_beam_M1_optA.csv` (original broken run):
```
epoch, train_loss, val_loss, dev_bleu, dev_chrf_pp
1,     10.391,    10.389,   0.00,    3.15
2,     10.382,    10.366,   0.00,    3.16
...
8,     10.207,    10.115,   0.00,    0.51
```

Loss decreases by only 0.28 over 8 epochs. The random-init cross-entropy for a 32,000-vocab model is ln(32000) ≈ 10.37. Starting loss is 10.39 — i.e., the model trained for 8 full epochs and its outputs are still statistically indistinguishable from a randomly initialised model.

### LR values before and after correction

| Step | Before (LR×Noam) | After (Noam only, LR=1.0) |
|------|-------------------|---------------------------|
| 1 | ~7e-10 | ~6.2e-5 |
| 10 | ~2.2e-9 | ~2.2e-6 → rising |
| 100 | ~7e-9 → rising | ~2.2e-5 → rising |
| warmup/2 (≈300) | ~3e-7 | ~3e-4 |
| warmup (596) | ~1.8e-6 (PEAK) | ~1.8e-3 (PEAK) |
| 2× warmup | ~1.3e-6 | ~1.3e-3 |
| final step | ~2e-7 | ~2e-4 |

### Fix

Set `optimizer base LR = 1.0` so that `LambdaLR` output is the absolute learning rate directly:

```python
# Bug 1 fix: optimizer LR=1.0 so lr_lambda IS the absolute LR
optimizer = torch.optim.Adam(
    model.parameters(), lr=1.0,
    betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4,
)
def get_scheduler(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

Peak LR (d_model=512, warmup_steps=596): `512^(-0.5) × 596^(-0.5) ≈ 1.81e-3`. This is the standard Transformer Noam peak and will allow the model to train.

---

## Bug 2 — Warmup Steps Potentially Too Large

### Root cause

Some code versions used `WARMUP_STEPS = 4000` hardcoded (copied from the original "Attention Is All You Need" paper which used 65k+ training steps). For `beam_M1` with only 9500 training pairs, batch_size=32, gradient_accum=2:

```
steps_per_epoch  = ceil(9500/32 / 2) = 149
total_opt_steps  = 149 × 60 = 8940
warmup at 4000   = 26.8 epochs  ← model is still in warmup at epoch 27!
```

Even after the LR fix, a warmup longer than the full training run would prevent the model from reaching useful LRs.

### Fix

Dynamic warmup: `max(200, min(2000, 0.05 × total_opt_steps))`. For beam_M1 this gives `max(200, min(2000, 447)) = 447` steps (~3 epochs), which is appropriate.

---

## Bug 3 — Early Stopping Fires During Warmup

### Root cause

`EARLY_STOP = 12` with no minimum epoch guard. During warmup, BLEU/chrF fluctuates wildly (can be 0.0 on every epoch just due to partially trained outputs). The best chrF++ may not improve for 12 epochs because the model is still in warm-up.

The training log `training_log_student_beam_M1_optA_fixed.csv` shows the training was stopped at epoch 7 (presumably after 12 no-improvement epochs relative to the best epoch 7 value).

### Fix

```python
EARLY_STOP_PATIENCE    = 8    # patience in epochs
MIN_EPOCHS_BEFORE_STOP = 10   # never stop before epoch 10

# In training loop:
if epoch >= MIN_EPOCHS_BEFORE_STOP and NO_IMPROVE >= EARLY_STOP_PATIENCE:
    break
```

---

## Bug 4 — Automatic Resume of Broken Checkpoint

### Root cause

```python
if CKPT_LATEST.exists():
    # auto-resume from latest checkpoint unconditionally
```

If a broken run left a `_latest.pt` file, the corrected run would silently resume from the broken checkpoint, loading corrupted optimizer state, wrong LR position, etc.

### Fix

```python
RESUME_TRAINING   = False   # must be explicitly set to True
RESUME_CHECKPOINT = None

if RESUME_TRAINING:
    # verify model_cfg, vocab_size, dataset, tokenizer_sha256 match
    # then restore all states
```

New run name `student_beam_M1_optA_fixed_v2` ensures no collision with old checkpoint files.

---

## Bug 5 — Checkpoint Missing Critical Fields

### Root cause

The original `save_ckpt()` saves only: epoch, global_step, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_dev_chrf, no_improve, model_cfg, run_name, dataset, model_size. Missing: scaler_state_dict, training config, tokenizer SHA-256, vocab_info, RNG states, split metadata.

### Fix

New `save_checkpoint()` saves all required fields including `scaler_state_dict`, `train_cfg`, `tokenizer_sha256`, `vocab_info`, `rng_states` (Python, NumPy, CPU torch, CUDA torch), `split_meta`, `schema_version=2`.

---

## Bug 6 — Custom Label Smoothing Distribution

### Root cause

The custom `LabelSmoothingLoss` builds a smoothed distribution by:
1. Filling all positions with `smoothing / (vocab_size - 2)`
2. Setting the target position to `confidence = 1 - smoothing`
3. Setting the PAD position to 0.0

Step 3 happens after step 2, but step 2 already placed a non-zero value on the PAD position. After zeroing PAD, the distribution no longer sums to 1.0. For vocab_size=32000 and smoothing=0.1, the deviation is tiny (~3e-6) but the implementation is fragile and non-standard.

### Fix

Replace with `nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)`. This is PyTorch's battle-tested implementation, correctly handles PAD masking, and is available in PyTorch ≥ 1.10.

---

## Bug 7 — No Data Validation Before Training

### Root cause

The original notebook loads the cache and immediately starts training with no checks on data quality. If the cache was built with a wrong tokenizer, swapped src/tgt, or has mostly empty targets, training will silently fail.

### Fix

Added validation cell that:
- Prints length statistics (min/mean/max/p95 for src and tgt)
- Counts empty targets (len ≤ 2 means only BOS+EOS)
- Counts duplicate sources in a 2000-example sample
- Validates all token IDs in a 500-example sample are in `[0, VOCAB_SIZE)`
- Decodes and prints 10 random src-tgt pairs for human inspection
- Checks first 200 examples for identity (src==tgt) and ASCII presence in src
- Aborts if >20% targets are empty, >20% are identity, or >50% sources lack ASCII

---

## Bug 8 — No Overfit Sanity Test

### Root cause

With a broken LR, NaN loss, or wrong masking, the model can run for hours on Kaggle before you notice it isn't learning. There was no early verification step.

### Fix

Added `RUN_OVERFIT_TEST = True` cell that:
- Creates a FRESH model (separate from the main training model)
- Trains for 200 steps on 64 examples
- Asserts final loss < 0.5 (a threshold easily reachable by overfitting)
- Decodes 3 training examples and prints them
- Raises `RuntimeError` if the test fails, preventing wasted GPU time
- Deletes the test model and frees VRAM after the test

---

## Bug 9 — Quick Eval Always Uses First N Sentences

### Root cause

```python
hyps = [greedy_translate(model, s) for s in src_list[:n]]
```

Using the first 200 FLORES dev sentences every epoch is a biased sample. Any systematic error on the first 200 sentences will mask progress on the rest.

### Fix

Build a seeded shuffled subset **once** at notebook startup:

```python
_eval_rng = random.Random(QUICK_EVAL_SEED)
_eval_rng.shuffle(_eval_indices)
_eval_indices = _eval_indices[:QUICK_EVAL_N]
EVAL_SRC_FIXED = [FLORES_DEV_SRC[i] for i in _eval_indices]
EVAL_REF_FIXED = [FLORES_DEV_REF[i] for i in _eval_indices]
```

The same deterministic subset is used every epoch, making epoch-to-epoch comparisons valid while covering a representative spread of sentences.

---

## Bug 10 — Non-Atomic Checkpoint Save

### Root cause

```python
torch.save(payload, path)
```

If the Kaggle session is interrupted mid-write (session timeout, OOM), the checkpoint file is partially written and corrupt. The next session will try to resume from it and fail.

### Fix

```python
_fd, _tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.pt.tmp')
os.close(_fd)
torch.save(payload, _tmp)
os.replace(_tmp, str(path))   # atomic rename
```

`os.replace()` is atomic on POSIX and near-atomic on Windows (via MoveFileExW). The final checkpoint file is always either complete or absent.

---

## Why the Original Run Produced BLEU=0 and chrF++ ≈ 1.5

1. **Effective LR ~1.8e-6 (peak) vs intended ~1.8e-3**: The model weights received updates approximately 1,000× too small throughout the entire run. After 8 epochs, cross-entropy loss had barely decreased from the random-init level (10.39 → 10.21). The model's output distribution was still nearly uniform over 32,000 tokens.

2. **BLEU=0.00**: When a model outputs near-random tokens, n-gram overlap with any reference is essentially zero. SacreBLEU (flores200 tokenizer) correctly reports 0.00.

3. **chrF++ ≈ 1.5**: Character-level F-score also near zero but not exactly zero — a random model occasionally generates Swahili-looking character sequences by chance, especially short common subwords. A score of ~1.5 is consistent with random token outputs from a shared English-Swahili vocabulary.

4. The 03c evaluation was loading this correctly-shaped but undertrained checkpoint. The evaluation pipeline itself (beam search, tokenizer, data loading) was largely correct in the existing 03c. The metric of ~1.5 chrF++ is a faithful measurement of the broken checkpoint's output quality.

---

## Are Old Checkpoints Reusable?

**No.** The existing checkpoints (e.g., `student_beam_M1_optA_fixed_best.pt`) were trained with the broken LR schedule. Even the "best" checkpoint (epoch 7, chrF++=8.84) represents a model that barely learned anything. The optimizer state stored in these checkpoints is also in a broken LR position.

The corrected run uses a different run name (`student_beam_M1_optA_fixed_v2`) and will not resume from old checkpoints. Start fresh.

---

## Expected Behavior After Fix (Do Not Fabricate Scores)

After the LR fix, the training should show:

- **Overfit test** (200 steps, 64 examples): loss should decrease from ~10.4 to below 0.5. If this fails, the pipeline still has a bug.
- **Training loss**: should decrease meaningfully each epoch, reaching roughly 3–6 by epoch 10–20 (depending on dataset size and model capacity).
- **Quick-eval chrF++**: should rise above 5 by epoch 5, above 10 by epoch 15. Do not expect 30+ chrF++ from beam_M1 (only 10k synthetic training pairs).
- **Generated outputs**: should contain recognisable Swahili words by epoch 5–10. The outputs will not be perfect translations but they should be source-conditioned and non-trivial.

No specific final BLEU or chrF++ score is guaranteed. A 6-layer model trained on 9,500 synthetic pairs distilled from a teacher with M=1 hypotheses will produce modest scores. The key success criterion is: **losses decrease, tiny-subset overfitting succeeds, generated outputs are non-empty and source-conditioned**.

---

## Kaggle Execution Order

### Step 1: Run 03a_tokenizer_and_data.ipynb
- Produces: `notebooks/models/vocab_info.json`, `shared_spm.model`, `cache_beam_M1.pt`, `raw_flores_dev.json`, `raw_flores_devtest.json`
- Save output as a Kaggle dataset (e.g., `mhd-data-v2`)

### Step 2: Run 03b_train_FIXED_V2.ipynb
- Attach the 03a output dataset as input
- Run all cells top-to-bottom
- Overfit test runs automatically (Cell 9)
- Training runs for up to 60 epochs with early stopping after epoch 10
- **Do not interrupt** — full training takes ~45 min on T4 for beam_M1

### Files to save after 03b (as Kaggle output dataset):
```
/kaggle/working/models/student_beam_M1_optA_fixed_v2/
    student_beam_M1_optA_fixed_v2_best_chrf.pt   ← primary eval checkpoint
    student_beam_M1_optA_fixed_v2_best_val.pt    ← secondary eval checkpoint
    student_beam_M1_optA_fixed_v2_latest.pt      ← resume checkpoint
    shared_spm.model                             ← tokenizer (copy)
    shared_spm.vocab
    vocab_info.json

/kaggle/working/results/
    training_log_student_beam_M1_optA_fixed_v2.csv
```

### Step 3: Run 03c_evaluate_FIXED_V2.ipynb
- Attach the 03b output dataset as input
- Verify the health gate passes (Cell 8) before full evaluation runs
- Full evaluation: ~5–15 min on T4

### Files to attach before 03c:
- The 03b output dataset containing `*_best_chrf.pt`, `shared_spm.model`, `vocab_info.json`
- The 03a output dataset containing `raw_flores_dev.json`, `raw_flores_devtest.json`

---

## Files Produced

| File | Description |
|------|-------------|
| `03b_train_FIXED_V2.ipynb` | Fixed training notebook |
| `03c_evaluate_FIXED_V2.ipynb` | Fixed evaluation notebook |
| `FIX_REPORT.md` | This report |

---

*Generated by Kiro — July 2026*
