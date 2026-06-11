from dotenv import load_dotenv
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

load_dotenv()

source = "https://www.youtube.com/watch?v=_dPv69PR7cM"
language = "hinglish"
chunks = process_input(source)
transcript = transcribe_all(chunks, language)
print("\n=== Transcript ===\n")
print(transcript)
