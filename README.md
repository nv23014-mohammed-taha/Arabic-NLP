# Arabic NLP Benchmark

A systematic evaluation of Arabic language models across multiple downstream NLP tasks.  
This benchmark compares Arabic-specific models (AraBERT, CAMeLBERT, MARBERT) against multilingual baselines (mBERT, XLM-R) on sentiment analysis, named entity recognition, and question answering.

---

##  Results

### Sentiment Analysis — HARD Dataset

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Precision | Recall | Speed (s/s) |
|---|---|---|---|---|---|---|
| **CAMeLBERT-DA** | **88.4** | **87.9** | **88.2** | 88.1 | 87.7 | 281 |
| MARBERTv2 | 87.1 | 86.5 | 87.0 | 86.8 | 86.2 | 288 |
| CAMeLBERT-MSA | 85.3 | 84.8 | 85.1 | 85.0 | 84.6 | 283 |
| AraBERT | 84.7 | 84.1 | 84.5 | 84.3 | 83.9 | 274 |
| XLM-R Base | 82.2 | 81.6 | 82.0 | 81.8 | 81.4 | 251 |
| mBERT | 78.6 | 77.9 | 78.4 | 78.2 | 77.6 | 302 |

> Full results across all tasks and datasets: [`results/`](results/)

---

##  Key Findings

1. **Dialectal pre-training matters** — CAMeLBERT-DA outperforms CAMeLBERT-MSA by ~3 F1 points on HARD (hotel reviews), which are written in Gulf/Egyptian dialect. This directly validates the importance of domain-matched pre-training.

2. **Social-media pre-training generalises** — MARBERTv2, trained entirely on tweets, achieves the second-best score despite HARD not being a Twitter dataset. Pre-training data volume and Arabic variety coverage appear to matter more than exact domain match.

3. **Multilingual models underperform Arabic-specific ones** — XLM-R base trails CAMeLBERT-DA by 6.3 F1 points, confirming that Arabic-specific pre-training provides a substantial benefit that cross-lingual transfer cannot fully recover.

4. **mBERT is a weak baseline** — Its cased multilingual training does not handle Arabic morphological complexity well; 9 F1 points behind the best model. XLM-R should be preferred as a multilingual baseline.

5. **No single model dominates all tasks** — MARBERT leads on social-media NER (dialectal tweets) while CAMeLBERT-MSA is stronger on formal-register QA.

---

##  Repository Structure

```
arabic-nlp-benchmark/
├── README.md               ← This file
├── data/
│   └── download.sh         ← Dataset download script
├── models/
│   ├── base.py             ← Abstract base class
│   ├── arabert.py          ← AraBERT wrapper
│   ├── camelbert.py        ← CAMeLBERT (MSA/DA/mixed) wrapper
│   ├── marbert.py          ← MARBERTv2 wrapper
│   └── multilingual_bert.py← mBERT + XLM-R wrappers
├── evaluate.py             ← Unified evaluation script (Hydra)
├── configs/
│   ├── sentiment.yaml      ← Sentiment task config
│   └── ner.yaml            ← NER task config
├── results/
│   ├── results.csv         ← Aggregated scores
│   └── plots/              ← Bar charts + confusion matrices
└── requirements.txt
```

---

##  Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/arabic-nlp-benchmark.git
cd arabic-nlp-benchmark
pip install -r requirements.txt
```

### 2. Download datasets

```bash
cd data && bash download.sh && cd ..
```

### 3. Run evaluation

```bash
# Sentiment (HARD dataset, all models)
python evaluate.py --config-name sentiment

# With a subsample for quick testing
python evaluate.py --config-name sentiment dataset.max_samples=200

# NER task
python evaluate.py --config-name ner

# Enable W&B logging
python evaluate.py --config-name sentiment output.log_wandb=true
```

Results are saved to `results/sentiment_results.csv` and plots to `results/plots/`.

---

##  Datasets

| Dataset | Task | Size | Source |
|---|---|---|---|
| [HARD](https://github.com/elnagara/HARD-Arabic-Dataset) | Sentiment | 93,700 reviews | Hotel reviews (Gulf Arabic) |
| [ArSAS](https://homepages.inf.ed.ac.uk/wmagdy/resources.htm) | Sentiment | 21,000 tweets | Twitter (MSA + DA) |
| [ANERcorp](https://camel.abudhabi.nyu.edu/anercorp/) | NER | 150,286 tokens | News (MSA) |
| [TyDi QA](https://ai.google.com/research/tydiqa) | QA | 14,805 passages | Wikipedia (MSA) |

---

## Models

| Model | Pre-training Data | Arabic Variety | Params |
|---|---|---|---|
| [AraBERT v2](https://huggingface.co/aubmindlab/bert-base-arabertv2) | 77GB news + Wikipedia | MSA | 136M |
| [CAMeLBERT-MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) | 107GB MSA text | MSA | 163M |
| [CAMeLBERT-DA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da) | 54GB dialectal | Dialectal | 163M |
| [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) | 1B dialectal tweets | Dialectal | 163M |
| [mBERT](https://huggingface.co/bert-base-multilingual-cased) | 104 languages | Multi | 179M |
| [XLM-R Base](https://huggingface.co/xlm-roberta-base) | 100 languages (CC-100) | Multi | 278M |

---

##  Configuration

This project uses [Hydra](https://hydra.cc/) for configuration. Every parameter is overridable from the command line:

```bash
# Change dataset
python evaluate.py --config-name sentiment dataset.name=arsas

# Disable specific models
python evaluate.py --config-name sentiment models.xlmr_large.enabled=false

# Change batch size
python evaluate.py --config-name sentiment evaluation.batch_size=64
```

---

##  Error Analysis

A qualitative look at failure cases for the best model (CAMeLBERT-DA) on sentiment:

**False Negatives (predicted positive, actually negative):**
- Reviews with sarcasm: *"الخدمة ممتازة... لو كنت في القرن الماضي"* (The service is excellent... if you were in the last century)
- Mixed-sentiment reviews: mostly positive with one negative detail

**False Positives (predicted negative, actually positive):**
- Reviews with religious expressions: *"الله يستر"* used positively
- Understatement common in Gulf dialect

**Takeaway:** Dialectal pragmatics and sarcasm remain unsolved challenges for current Arabic NLP models.

---

##  What I Learned

- Arabic NLP is not one problem — MSA, Classical, and Dialectal varieties behave very differently and warrant separate models.
- The gap between Arabic-specific and multilingual models is larger than I expected (~6–10 F1 points), suggesting that cross-lingual transfer still struggles with Arabic's morphological complexity.
- Pre-training data *variety* (which Arabic dialects are covered) matters as much as data *quantity*.
- Evaluation is hard: label consistency across Arabic NLP datasets is inconsistent, and some "gold" labels show annotator disagreement >20%.

---

##  References

```bibtex
@inproceedings{antoun2020arabert,
  title={AraBERT: Transformer-based Model for Arabic Language Understanding},
  author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
  booktitle={LREC 2020 Workshop},
  year={2020}
}

@article{inoue2021camelbert,
  title={CAMeLBERT: A collection of pre-trained models for Arabic NLP},
  author={Inoue, Go and others},
  journal={arXiv:2103.06678},
  year={2021}
}

@inproceedings{abdul2021marbert,
  title={ARBERT \& MARBERT: Deep Bidirectional Transformers for Arabic},
  author={Abdul-Mageed, Muhammad and others},
  booktitle={ACL 2021},
  year={2021}
}
```

---

##  License

MIT License. Dataset licenses vary — see each dataset's original source.

---

*Part of a series of ML research projects. See also:*  
*→ [Efficient Transformers](../efficient-transformers) | [Scaling Laws Experiment](../scaling-laws-experiment)*
