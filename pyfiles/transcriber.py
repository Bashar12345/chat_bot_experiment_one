import os
import pyaudio
import numpy as np
import torch
from scipy.signal import resample
from pynput import keyboard
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Suppress TensorFlow and other warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configuration
MODEL_NAME = "openai/whisper-tiny"  # Use a smaller model for faster transcription
FORMAT = pyaudio.paInt16
CHUNK = 2048  # Increased buffer size to prevent overflow
DEVICE_INDEX = 0  # Replace with the correct device index
RATE = 44100  # Default sample rate of your device
TARGET_RATE = 16000  # Whisper's required sample rate
CHANNELS = 1  # Use mono input for transcription

# Load Whisper model and processor
print("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

print(f"Using device '{audio.get_device_info_by_index(DEVICE_INDEX)['name']}' at {RATE} Hz.")
print("Recording and transcribing in real-time... Press 'S' to stop.")

# Real-time buffer
audio_buffer = np.zeros((TARGET_RATE * 5,), dtype=np.int16)  # 5-second sliding window
recording = True

# Handle key press to stop transcription
def on_press(key):
    global recording
    try:
        if key.char == 's':  # Stop on 'S' key press
            print("\n'S' key pressed. Stopping transcription...")
            recording = False
            return False
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    while recording:
        try:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_samples = np.frombuffer(data, dtype=np.int16)

            # Resample from 44100 Hz to 16000 Hz
            resampled_audio = resample(audio_samples, int(len(audio_samples) * TARGET_RATE / RATE))

            # Add to audio buffer
            audio_buffer = np.concatenate((audio_buffer, resampled_audio))[-TARGET_RATE * 5:]

            # Normalize audio and process with Whisper
            input_features = processor(
                audio_buffer.astype(np.float32) / 32768.0,  # Normalize int16 to float32
                sampling_rate=TARGET_RATE,
                return_tensors="pt",
                language="en"  # Set language for transcription
            ).input_features

            input_features = input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Print the transcription
            print(f"\rTranscription: {transcription}", end="")
        except OSError as e:
            if e.errno == -9981:  # Input overflow error
                print("\nWarning: Input overflowed. Skipping this chunk.")
                continue
except Exception as e:
    print(f"\nError: {e}")
finally:
    # Cleanup
    stream.stop_stream()
    stream.close()
    audio.terminate()
    listener.join()
    print("\nTranscription stopped.")
