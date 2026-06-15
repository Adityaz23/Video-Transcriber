from dotenv import load_dotenv
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_question_decision,
)

load_dotenv()

source = "https://www.youtube.com/watch?v=zfYsSFY4l18"
language = "english"

chunks = process_input(source)

transcript = transcribe_all(chunks, language)
print(f"\n" + "=" * 60)
print("\n=== Transcript ===\n")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 5000 else transcript)

title = generate_title(transcript)
summary = summarize(transcript)

print("\n" + "=" * 60)
print(f"TITLE: {title}")
print("=" * 60)
print("-" * 60)
print(summary)

action_items = extract_action_items(transcript)
decision = extract_key_decisions(transcript)
questions = extract_question_decision(transcript)

print("\n" + "=" * 60)
print("ACTION ITEMS")
print("=" * 60)
print(action_items)

print("\n" + "=" * 60)
print("KEY DECISIONS")
print("=" * 60)
print(decision)

print("\n" + "=" * 60)
print("OPEN QUESTIONS")
print("=" * 60)
print(questions)
