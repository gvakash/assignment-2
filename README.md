# Assignment 2 – Forced Alignment and Multimodal Emotion Recognition

This repo has two separate pieces of work. One uses the Montreal Forced Aligner (MFA) to line up English recordings with their transcripts. The other trains speech, text and fusion models to recognise emotion in the TESS dataset. They don't share any code or data, so each folder can be used on its own.


## MFA

Runs MFA on six recordings with the stock `english_us_arpa` dictionary. With the stock dictionary MFA gives these words a single `spn` ("spoken noise") segment instead of phones. With the custom entries every one of them gets a proper phone sequence.

Includes the corpus preparation, OOV finding and comparison scripts, both sets of TextGrids and the before/after figures. See [MFA/README.md](MFA/README.md).

## Multimodal Emotion Recognition

Trains three kinds of models speech only (CNN + BiGRU with attention), text only (character CNN + BiGRU) and fusion of the two (concatenation or a learned gate) on the TESS emotional speech dataset and compares them. Every model is trained with three seeds.

The test words are never seen in training, but the two speakers are, so the speech results are high. The text model sits at chance because every TESS sentence is "say the word X", which carries no emotion. An extra experiment trains on one speaker and tests on the other: accuracy drops to 44–62%, and fusion is not better than speech alone.

Includes all scripts, trained checkpoints, logs, tables and figures. See [Multimodal Emotion Recognition/README.md](Multimodal%20Emotion%20Recognition/README.md).


## References

- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (McAuliffe et al., 2017)
- [TESS: Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487) (Dupuis & Pichora-Fagerstrom, 2010)

