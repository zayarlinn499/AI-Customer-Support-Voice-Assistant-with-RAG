from dotenv import load_dotenv
import os

from elevenlabs.client import ElevenLabs

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

audio = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    text="The refund period is 30 calendar days.",
    model_id="eleven_multilingual_v2"
)

with open("response.mp3", "wb") as f:
    for chunk in audio:
        f.write(chunk)

print("Audio saved as response.mp3")