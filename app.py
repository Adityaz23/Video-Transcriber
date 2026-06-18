import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all_chunks
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_question_decision,
    extract_key_decisions,
)
from core.rag_engine import ask_question, build_rag_chain

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False


def reset_session():
    st.session_state.result = None
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Pipeline runner with step-by-step status updates
# ---------------------------------------------------------------------------
def run_pipeline_with_status(source: str, language: str) -> dict:
    with st.status(
        "Running the AI Video Assistant pipeline...", expanded=True
    ) as status:
        st.write("📥 Processing input and extracting/chunking audio...")
        chunks = process_input(source)
        st.write(f"✅ Created {len(chunks)} audio chunk(s).")

        st.write(
            f"🎙️ Transcribing audio (engine: {'Sarvam AI' if language == 'hinglish' else 'Whisper'})..."
        )
        transcript = transcribe_all_chunks(chunks, language=language)
        st.write("✅ Transcription complete.")

        st.write("🏷️ Generating title...")
        title = generate_title(transcript)

        st.write("📝 Generating summary...")
        summary = summarize(transcript)

        st.write("✅ Extracting action items...")
        action_items = extract_action_items(transcript)

        st.write("📌 Extracting key decisions...")
        decisions = extract_key_decisions(transcript)

        st.write("❓ Extracting open questions...")
        questions = extract_question_decision(transcript)

        st.write("🔍 Building RAG chain for Q&A...")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Pipeline complete!", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------------------------------------------------------
# Sidebar — input controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎥 AI Video Assistant")
    st.caption(
        "Turn any meeting video/audio into a transcript, summary, and a chat-ready knowledge base."
    )

    st.divider()

    source_type = st.radio(
        "Source type", ["YouTube URL", "Local file path"], horizontal=False
    )

    if source_type == "YouTube URL":
        source_input = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
        )
    else:
        source_input = st.text_input(
            "Local file path", placeholder="/path/to/video_or_audio.mp4"
        )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_clicked = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if st.session_state.result is not None:
        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            reset_session()
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("AI Video Assistant")

if run_clicked:
    if not source_input or not source_input.strip():
        st.error("Please provide a YouTube URL or a local file path.")
    else:
        try:
            st.session_state.result = run_pipeline_with_status(
                source_input.strip(), language
            )
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.session_state.result = None

result = st.session_state.result

if result is None:
    st.info(
        "Enter a YouTube URL or local file path in the sidebar, then click **Run Pipeline** to get started."
    )
else:
    st.subheader(result["title"])

    tab_summary, tab_actions, tab_transcript, tab_chat = st.tabs(
        ["📝 Summary", "✅ Action Items & Decisions", "📄 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(result["summary"])

    with tab_actions:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Action Items")
            st.markdown(result["action_items"])
            st.markdown("#### ❓ Open Questions")
            st.markdown(result["open_questions"])
        with col2:
            st.markdown("#### 📌 Key Decisions")
            st.markdown(result["key_decisions"])

    with tab_transcript:
        st.text_area("Full transcript", result["transcript"], height=500)
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

    with tab_chat:
        st.caption(
            "Ask questions about the meeting — answers are grounded in the transcript via RAG."
        )

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        question = st.chat_input("Ask something about this meeting...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"Error while answering: {e}"
                st.markdown(answer)

            st.session_state.chat_history.append(("assistant", answer))
