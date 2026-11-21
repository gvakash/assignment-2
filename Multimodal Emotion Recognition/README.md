# Multimodal Emotion Recognition on TESS

This project trains three kinds of emotion classifier(speech only, text only and fusion of both) on the TESS dataset and compares them. All three use a temporal block (CNN) followed by a contextual block (BiGRU with attention pooling). It is small enough to train on a CPU in a few minutes per model.

Dataset: [TESS, Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487) (Dupuis & Pichora-Fagerstrom, 2010)


## How to re-implement?

### Dependencies

```bash
pip install -r requirements.txt
```

### Evaluate the trained models

The trained checkpoints (about 0.4–1.2 MB each) are in `models/*/checkpoints/`. The spectrogram features and audio are not in the repo, so first rebuild them (next section), then:

```bash
python models/speech_pipeline/test.py
python models/text_pipeline/test.py
python models/fusion_pipeline/test.py --mode concat
python models/fusion_pipeline/test.py --mode gated
python analysis.py --fusion concat --seed 0
```

Each test script runs seeds 0, 1 and 2 by default (pass `--seeds` to choose others) and writes a confusion matrix, `*_misclassified.csv`, `*_results.json` and the test embeddings to `Results/`. `analysis.py` reads the `.json` files and embeddings to build the accuracy table, per-emotion recall, failure cases and the t-SNE cluster plot.

## Reproducing everything

Run these from this folder: the first three scripts use relative paths (`data/`, `metadata.csv`, `features/`).

### 1. Data

Download TESS and unzip it under `data/`, so the folders look like `data/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav`. TESS clips come at 24,414 Hz or 96,000 Hz; they are resampled to 16 kHz in the next steps.

### 2. Metadata, splits and features

```bash
python make_metadata.py data        # builds metadata.csv (speaker, word, emotion, text, duration)
python make_splits.py               # adds the word-disjoint train/val/test column (seed 42)
python extract_features.py          # log-mel features -> features/speech_logmel.pt, example plot
```

`metadata.csv` is already in the repo with the splits filled in. Re-running `make_splits.py` gives the same split, because the shuffle is seeded.

### 3. Training

```bash
for s in 0 1 2; do python models/speech_pipeline/train.py --seed $s; done
for s in 0 1 2; do python models/text_pipeline/train.py   --seed $s; done
for s in 0 1 2; do python models/fusion_pipeline/train.py --seed $s --mode concat; done
for s in 0 1 2; do python models/fusion_pipeline/train.py --seed $s --mode gated;  done
```

### 4. Cross-speaker experiment

```bash
python cross_speaker.py --seeds 0 1 2
python cross_failures.py
```

A word-disjoint slice of the *training* actress (30 words) is used for early stopping, so the test actress is never used during training. `cross_failures.py` picks the 5 most common error patterns and draws each with the most confidently wrong example.

## How the models work

- **Speech encoder** ([common.py](common.py)): three Conv2d + BatchNorm + ReLU + MaxPool blocks (temporal block), a linear projection, then a BiGRU with attention pooling (contextual block).
- **Text encoder** ([text_common.py](text_common.py)): characters (a–z, space, apostrophe, hyphen) are embedded and passed through two Conv1d layers (letter n-grams), then the same BiGRU + attention. Character input means words never seen in training still work.
- **Fusion** ([fusion_common.py](fusion_common.py)): the two contextual vectors are combined either by concatenation followed by Linear + ReLU, or by a learned gate, `g·speech + (1−g)·text`.
- **Checkpoints** store the weights, the mel mean/std, the seed, the best epoch and its validation accuracy.





## References

- [TESS: Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487)
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) (Park et al., 2019)
- PyTorch, torchaudio and scikit-learn

