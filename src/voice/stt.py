from faster_whisper import WhisperModel

print("Loading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Model loaded!")

segments, info = model.transcribe(
    "audio/test.wav"
)

print("\nDetected Language:")
print(info.language)

print("\nTranscription:")

for segment in segments:
    print(segment.text)