from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import (
    transcribe_all_chunks,
)
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_question_decision,
    extract_key_decisions,
)
from core.rag_engine import ask_question, build_rag_chain

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting the AI Video Assistant")
    chunks = process_input(source)
    transcript = transcribe_all_chunks(chunks, language=language)
    print(f"raw transcript (first 300 characters) {transcript[:300]}")
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_item = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_question_decision(transcript)
    rag_chain = build_rag_chain(transcript)
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter youtube url or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    result = run_pipeline(source, language)

print("\n" + "=" * 60)
print(f"📌 Title: {result['title']}")
print("\n📝 Summary:")
print(result["summary"])

print("\n📋 Action Items:")
if result["action_items"]:
    print(result["action_items"])
else:
    print("No action items were identified.")

print("\n🎯 Key Decisions:")
if result["key_decisions"]:
    print(result["key_decisions"])
else:
    print("No key decisions were detected.")

print("\n❓ Open Questions:")
if result["open_questions"]:
    print(result["open_questions"])
else:
    print("No unanswered questions were found.")

print("\n" + "=" * 60)
print("💬 Interactive Q&A Session")
print("Ask anything about the video/meeting.")
print("Type 'exit' to quit.\n")
