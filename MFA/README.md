# Montreal Forced Aligner 

This project aligns six English recordings with their transcripts using the Montreal Forced Aligner (MFA). The goal is to see what happens to words the dictionary doesn't know(OOV words), and whether adding them fixes the alignment. I've used MFA 3.4.4 pinned in environment.yml

MFA: [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (McAuliffe et al., 2017)

## How to re-implement?

### 1. Install MFA

MFA only installs through conda:

```bash
conda env create -f environment.yml      # creates the "mfa" env with MFA 3.4.4
conda activate mfa
mfa version                              # should print 3.4.4

# pretrained English (US, ARPAbet) acoustic model, dictionary and G2P model
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

The scripts in `scripts/` use only the Python standard library, so no extra pip install is needed.

### 2. Prepare the dataset

MFA needs each audio file next to a transcript with the same name. The original recordings are in `raw/`: a `.wav` (NIST SPHERE format, which MFA reads directly) and a `.txt` transcript for each of the six recordings (`F2BJRLP1–3` and `ISLE_SESS0131_BLOCKD02_01–03_sprt1`). The transcripts have line breaks and hyphen breaks like "seventy- six" in them.

```bash
python scripts/prepare_corpus.py raw corpus
```

This copies each `.wav` into `corpus/` and writes a matching `.lab` file: the transcript on one line, uppercased, with the line breaks and hyphen breaks removed. For example `corpus/F2BJRLP1.wav` + `corpus/F2BJRLP1.lab`. MFA reads `corpus/` as its input.

### 3. Run the alignment

```bash
# Check the corpus against the stock model and dictionary
mfa validate corpus english_us_arpa english_us_arpa --single_speaker --clean

# Baseline alignment with the stock dictionary
mfa align corpus english_us_arpa english_us_arpa output_baseline --single_speaker --clean

# Find the words the stock dictionary doesn't have, and get G2P guesses for them
python scripts/find_oovs.py corpus ~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict | tee oov_list.txt
mfa g2p corpus english_us_arpa oov_g2p.txt --dictionary_path english_us_arpa
python scripts/list_spn.py output_baseline | tee spn_baseline.txt

# Build the custom dictionary: stock entries + dictionary/custom_entries.dict
cat ~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict \
    dictionary/custom_entries.dict > dictionary/english_us_arpa_custom.dict

# Align again with the custom dictionary
mfa align corpus dictionary/english_us_arpa_custom.dict english_us_arpa output_custom --single_speaker --clean

# Compare the two runs and list any remaining spn segments
python scripts/list_spn.py output_custom | tee spn_custom.txt
python scripts/compare_runs.py output_baseline output_custom | tee comparison.txt
```

`--single_speaker` treats the whole corpus as one speaker, and `--clean` removes MFA's cached files from earlier runs. The dictionary path in the `find_oovs.py` and `cat` commands depends on where MFA stores its models on your machine (`mfa model inspect` or `MFA_ROOT_DIR` will tell you). The merged `english_us_arpa_custom.dict` is not committed, because it is about 200k lines and can be rebuilt with the `cat` command.

### 4. Outputs

Each `mfa align` run writes one **TextGrid file** per recording, plus an `alignment_analysis.csv`:

| Folder | Dictionary | Contents |
|---|---|---|
| `output_baseline/` | stock `english_us_arpa` | 6 TextGrids + `alignment_analysis.csv` |
| `output_custom/` | stock + 5 custom entries | 6 TextGrids + `alignment_analysis.csv` |

Each TextGrid (e.g. `output_custom/F2BJRLP1.TextGrid`) has two interval tiers that you can open in [Praat](https://www.fon.hum.uva.nl/praat/):

- `words`: start and end time of every word
- `phones`: start and end time of every phone (ARPAbet with stress marks, such as `AA1`). Words with no pronunciation show up as one `spn` segment.

`alignment_analysis.csv` has one row per recording with the overall and speech log-likelihood, phone duration deviation and SNR.

The other files come from the helper scripts: `oov_list.txt`, `oov_g2p.txt`, `spn_baseline.txt`, `spn_custom.txt` (empty, since no `spn` is left) and `comparison.txt`.

## References

- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- MFA pretrained `english_us_arpa` acoustic model and dictionary
- ISLE learner corpus and the news-style recordings (`F2BJRLP`) used as input audio

