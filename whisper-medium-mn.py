# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="bayartsogt/whisper-medium-mn-10")

# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("bayartsogt/whisper-medium-mn-10")
model = AutoModelForSpeechSeq2Seq.from_pretrained("bayartsogt/whisper-medium-mn-10")