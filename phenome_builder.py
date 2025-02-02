import os
import torchaudio
from datasets import load_dataset, DatasetDict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import numpy as np
import torch
import nltk
from phonemizer import phonemize

nltk.data.path.append('/home/vai/Desktop/experiment/nltk_data')

nltk.download('cmudict', download_dir='/home/vai/Desktop/experiment/nltk_data')
nltk.download('punkt', download_dir='/home/vai/Desktop/experiment/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='/home/vai/Desktop/experiment/nltk_data')



try:
    nltk.data.find('taggers/averaged_perceptron_tagger/')
    print("The resource was found!")
except LookupError:
    print("The resource was NOT found.")



# Load the Wav2Vec2 processor and model (base model)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model is running on: {device}")


# Dataset Preparation
def preprocess_audio(batch):
    # Load and resample audio
    audio = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio = resampler(torch.tensor(audio)).numpy() 

    # Convert sentence to phonemes
    phonemes = phonemize(batch["sentence"], language='en-us', backend='espeak')
    phoneme_ids = processor.tokenizer(" ".join(phonemes), return_tensors="pt", padding=True, truncation=True, max_length=processor.tokenizer.model_max_length).input_ids[0]

    # Prepare input and labels
    batch["input_values"] = processor(audio, sampling_rate=16000, return_tensors="pt").input_values[0]
    batch["labels"] = phoneme_ids
    print("Model device:", next(model.parameters()).device)
    print("Input tensor device:", batch["input_values"].device)
    return batch


# Load dataset from Hugging Face Hub
data = load_dataset("DTU54DL/common-accent")

sentence_lengths = [len(ex["sentence"]) for ex in data["train"]]
print("Max length:", max(sentence_lengths))
print("Average length:", sum(sentence_lengths) / len(sentence_lengths))






if __name__ == "__main__":
    # Preprocess datasets
    data = data.map(preprocess_audio, remove_columns=["audio", "sentence"])


    # Define training arguments with mixed-precision training enabled
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4, 
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        fp16=True,  # Enable mixed-precision training for better GPU utilization
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=processor.feature_extractor,
    )

    # Train the model
    trainer.train()
    

