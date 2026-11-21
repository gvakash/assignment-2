# Montreal Forced Aligner 

This project aligns six English recordings with their transcripts using the Montreal Forced Aligner (MFA). The goal is to see what happens to words the dictionary doesn't know(OOV words), and whether adding them fixes the alignment. I've used MFA 3.4.4 pinned in environment.yml

MFA: [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (McAuliffe et al., 2017)

## How to re-implement?

### Dependencies

MFA only installs through conda:

```bash
conda env create -f environment.yml      # creates the "mfa" env with MFA 3.4.4
conda activate mfa
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

The scripts in `scripts/` use only the Python standard library.

### Run

The commands below are the standard MFA workflow for this folder's layout. The exact flags used for the checked-in outputs are not saved in the repo.

```bash
# 1. Turn raw/ (wav + transcript) into corpus/ (wav + .lab, uppercased and cleaned)
python scripts/prepare_corpus.py raw corpus

# 2. Find words the stock dictionary doesn't have
python scripts/find_oovs.py corpus ~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict

# 3. Baseline alignment with the stock dictionary
mfa align corpus english_us_arpa english_us_arpa output_baseline --clean

# 4. Build the custom dictionary: stock entries + dictionary/custom_entries.dict
cat ~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict \
    dictionary/custom_entries.dict > dictionary/english_us_arpa_custom.dict

# 5. Align again with the custom dictionary
mfa align corpus dictionary/english_us_arpa_custom.dict english_us_arpa output_custom --clean

# 6. Compare the two runs and list any remaining spn segments
python scripts/compare_runs.py output_baseline output_custom
python scripts/list_spn.py output_custom
```

The dictionary path in steps 2 and 4 depends on where MFA stores its models on your machine (`mfa model inspect` or `MFA_ROOT_DIR` will tell you). The merged `english_us_arpa_custom.dict` is not committed, because it is about 200k lines and can be rebuilt in step 4.



## References

- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- MFA pretrained `english_us_arpa` acoustic model and dictionary
- ISLE learner corpus and the news-style recordings (`F2BJRLP`) used as input audio

