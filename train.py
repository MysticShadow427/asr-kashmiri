from dataset import DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperTokenizer,WhisperProcessor,WhisperForConditionalGeneration,Seq2SeqTrainingArguments,Seq2SeqTrainer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Urdu", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Urdu", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.generation_config.language = "urdu"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
device = 'cuda'
model = model.to(device)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ks",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=3000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

model_save_path = "./ks-whisper"
trainer.model.save_pretrained(model_save_path)
trainer.tokenizer.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)
