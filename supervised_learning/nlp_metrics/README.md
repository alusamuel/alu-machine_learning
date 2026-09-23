# NLP Metrics

This directory contains exercises that compute BLEU (Bilingual Evaluation
Understudy) scores. BLEU measures how close a machine-generated sentence is
to one or more reference translations.

## Files

- `0-uni_bleu.py`: calculates the unigram BLEU score for a sentence.
- `1-ngram_bleu.py`: calculates the n-gram BLEU score for a sentence.
- `2-cumulative_bleu.py`: calculates the cumulative n-gram BLEU score, the
  geometric mean of the 1-gram to n-gram precisions.

## How the score works

1. **Clipped precision**: each n-gram in the sentence counts at most as many
   times as it appears in a single reference.
2. **Brevity penalty**: a sentence shorter than the closest reference length
   `r` is penalized by `exp(1 - r / c)`, where `c` is the sentence length.
3. **BLEU** = brevity penalty × precision.

## Example

```python
references = [["the", "cat", "is", "on", "the", "mat"],
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

uni_bleu(references, sentence)            # 0.6549846024623855
ngram_bleu(references, sentence, 2)       # 0.6140480648084865
cumulative_bleu(references, sentence, 4)  # 0.5475182535069453
```
