from datasets import Dataset, DatasetDict
import pandas as pd
import torchaudio
from transformers import WhisperFeatureExtractor
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

train_df = pd.read_csv('/content/drive/MyDrive/kashmiri_dataset/combined_csv_file.csv')
val_df = pd.read_csv('/content/drive/MyDrive/kashmiri_dataset/combined_csv_valid.csv')
train_df = train_df.drop(columns=['start','end'],axis = 1)
val_df = val_df.drop(columns=['start','end'],axis = 1)
train_df = train_df.rename(columns={'filename_segment':'sentence'})
val_df = val_df.rename(columns={'filename_segment':'sentence'})


def load_audio(row):
    waveform, sample_rate = torchaudio.load(row['path'])
    return {'audio': {'array': waveform.squeeze(0).tolist(), 'sampling_rate': sample_rate}}

def df_to_hf_dataset(df):

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(load_audio, num_proc=1)
    return dataset

dataset_dict = DatasetDict({
    'train': df_to_hf_dataset(train_df),
    'test': df_to_hf_dataset(val_df),
})

print(dataset_dict)

dataset_dict = dataset_dict.remove_columns(["path"])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):

    audio = batch["audio"]

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

dataset_dict = dataset_dict.map(prepare_dataset, remove_columns=dataset_dict.column_names["train"], num_proc=1)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
