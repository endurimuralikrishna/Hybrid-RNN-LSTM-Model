# Hybrid RNN-LSTM: reconstructed reproducibility package
**Status:** reconstructed from the manuscript, not original experimental code; not validated against reported results. Do not deposit as original code. Run, verify and disclose differences before publication.

## Setup
Use Python 3.11 in a virtual environment. Install `pip install -r requirements.txt`. Obtain GoogleNews Word2Vec 300-dimensional binary separately from an authorized source. Do not redistribute pretrained weights without permission. BERT downloads its own pretrained checkpoint.

## Run
After verifying permissions, configure Kaggle credentials and run `python download_data.py --confirm-permission` (or obtain the CSV manually). Inspect its column names and label definitions. Example command (replace names and labels with verified values):

```
python run.py --csv data/raw/dataset.csv --text-column text --label-column label --positive-label 1 --negative-label 0 --word2vec /path/GoogleNews-vectors-negative300.bin --models lr svm lstm bilstm cnn bert rnn_lstm
```

Outputs: aggregate metrics, confusion matrices, training histories, plots, dataset counts and input checksum. No raw posts are written to outputs. The supplied example schema is not training data.

## Method and deviations
Manuscript specifies Word2Vec 300, sequence length 100, 64 RNN and LSTM units, dropout 0.3, batch 32, Adam, binary cross-entropy, 10 epochs and 80:20 train/test split. Reconstruction uses seed 42, a separate validation split from training, stratification, frozen embeddings, normalized-text deduplication and removal of conflicting labels. Cleaning, baseline hyperparameters, embedding handling, validation strategy and BERT fine-tuning are reconstruction decisions, not verified original settings. No stemming/lemmatization or stopword removal is applied because exact original procedure is unknown. No user-level split is possible without verified user identifiers. Results are post-label classification, not clinical diagnosis. Manuscript reports 7,731 rows and 7,650 unique posts but its confusion matrix totals 1,547; verify actual splits and rerun before reporting results. Do not substitute invented metrics.

## Deposit checklist
Replace or reconcile reconstructed scripts with original implementation; verify results and dataset permission; include environment and exact commands; publish code only after author approval; obtain DOI and update manuscript. Do not upload credentials, raw data, model weights or sensitive outputs without appropriate permission. Review code licensing with all authors before adding a license.
