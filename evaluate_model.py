import evaluate

metric = evaluate.load("wer")

def transcribe_audio(audio_path):

    speech_array, sampling_rate = torchaudio.load(audio_path)

    if sampling_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = transform(speech_array)
        sampling_rate = 16000

    input_features = processor(speech_array.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt").input_features.to('cuda')


    predicted_ids = model.generate(input_features,use_cache = False)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

audio_file = "/kashmiri_dataset/Segments/1970324836995208_segment_1.wav"


transcription = transcribe_audio(audio_file)
print("Transcription:", transcription)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

trainer.evaluate()
