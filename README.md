# Whisper Kashmiri Speech-to-Text

This repository contains scripts for training and evaluating an Urdu speech-to-text model using OpenAI's Whisper.

## Files Overview

### 1. `data_preprrocess.py`
- Extracts audio segments from `.wav` files using JSON metadata.
- Saves the extracted segments and metadata as a CSV file (`all_segments_info.csv`).

### 2. `dataset.py`
- Loads the segmented audio dataset and converts it into a Hugging Face `DatasetDict`.
- Applies `torchaudio` preprocessing and `WhisperFeatureExtractor`.
- Prepares the dataset for training by tokenizing text and extracting features.

### 3. `train.py`
- Loads the Whisper tokenizer, processor, and model (`openai/whisper-small`).
- Defines a data collator for batch processing.
- Configures training arguments and fine-tunes the model using `Seq2SeqTrainer`.
- Saves the trained model.

### 4. `evaluate_model.py`
- Loads the trained model to transcribe an audio file.
- Computes Word Error Rate (WER) for evaluation.
- Outputs the transcription and evaluation metrics.

---
## How to Train the Model

1. **Preprocess Data:**
   ```bash
   python data_preprrocess.py
   ```
   This generates segmented audio files and metadata in `all_segments_info.csv`.

2. **Prepare Dataset:**
   ```bash
   python dataset.py
   ```
   This converts the dataset into a Hugging Face format, ready for training.

3. **Train the Model:**
   ```bash
   python train.py
   ```
   This fine-tunes `openai/whisper-small` on the dataset and saves the trained model in `./ks-whisper`.

---
## How to Generate Transcriptions

After training, run:
```bash
python evaluate_model.py
```
This transcribes an example audio file and prints the transcription.

---
## Evaluation
The model is evaluated using **Word Error Rate (WER)**. The `evaluate_model.py` script computes WER for test predictions.

---
## Output
- Transcriptions for input audio files.
- Word Error Rate (WER) evaluation.
- Trained model stored in `./ks-whisper`.

This setup enables efficient Kashmiri speech-to-text conversion using OpenAI's Whisper model.

