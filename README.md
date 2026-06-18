# 🎥 AI Video Assistant (Video-Transcriber)

Turn any meeting video, YouTube link, or local audio/video file into a clean transcript, an AI-generated summary, extracted action items, key decisions, open questions — and a chat-ready knowledge base you can ask questions against using RAG (Retrieval-Augmented Generation).

The project ships with two interfaces:
- **`app.py`** — a Streamlit web UI with tabs for Summary, Action Items, Transcript, and Chat.
- **`main.py`** — a command-line pipeline for the same workflow.

## How it works

1. **Input ingestion** — Give it a YouTube URL or a local file path. YouTube audio is downloaded with `yt-dlp`; local files are converted to WAV with `pydub`/FFmpeg.
2. **Chunking** — The audio is split into 10-minute chunks for processing.
3. **Transcription** — Each chunk is transcribed:
   - **English** → local **Faster-Whisper** model (runs on CPU, no API key needed).
   - **Hinglish** → **Sarvam AI**'s speech-to-text-translate API (audio is further split into ≤25s pieces since Sarvam's sync API caps at 30s).
4. **Analysis (via Groq + LangChain)** — The transcript is summarized, titled, and mined for action items, key decisions, and open questions using `llama-3.3-70b-versatile` on Groq.
5. **RAG Q&A** — The transcript is embedded with a HuggingFace sentence-transformer model and stored in a local **Chroma** vector database, so you can chat with the transcript afterward.

## Project structure

```
Video-Transcriber/
├── app.py                     # Streamlit web app (main entry point)
├── main.py                    # CLI pipeline runner
├── test.py                    # Ad-hoc script for manual testing
├── requirements.txt
├── core/
│   ├── transcriber.py         # Whisper (local) + Sarvam AI transcription
│   ├── summarize.py           # Map-reduce summarization + title generation (Groq)
│   ├── extractor.py           # Action items / decisions / open questions (Groq)
│   ├── rag_engine.py          # Builds and queries the RAG chain
│   └── vector_store.py        # Chroma vector store + HuggingFace embeddings
├── utils/
│   └── audio_processor.py     # YouTube download, WAV conversion, chunking
└── vector_db/                 # Local Chroma persistence directory (auto-created)
```

## Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and available on your system `PATH` (required by `pydub` and `yt-dlp` for audio extraction/conversion)
- A **Groq API key** (free tier available) — used for summarization, title generation, action item/decision/question extraction, and RAG answers
- A **Sarvam AI API key** — only required if you plan to use the `hinglish` language option

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Adityaz23/Video-Transcriber.git
cd Video-Transcriber
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install FFmpeg

FFmpeg must be installed separately from the Python packages.

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

Verify it's on your PATH:

```bash
ffmpeg -version
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

Additionally, since the code uses **Faster-Whisper** and **Groq** directly (rather than `openai-whisper`/Mistral, which are listed in `requirements.txt` but unused by the current code), install these as well:

```bash
pip install faster-whisper langchain-groq langchain-chroma
```

> **Note:** `requirements.txt` in this repo references `openai-whisper`, `torch`, and `langchain-mistralai`, which were used in earlier versions of the project. The current code (`core/transcriber.py`, `core/summarize.py`, `core/extractor.py`, `core/rag_engine.py`) has since moved to **`faster-whisper`** for transcription and **Groq** (`langchain-groq`) for all LLM calls. You may see unused dependencies install, or missing-package errors for `faster_whisper`, `langchain_groq`, and `langchain_chroma` — install them manually as shown above if that happens.

### 5. Set up environment variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following keys:

```env
# Required — used for summarization, extraction, title generation, and RAG Q&A
GROQ_API_KEY=your_groq_api_key_here

# Required only if using the "hinglish" language option
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_SST_TRANSLATE_URL=https://api.sarvam.ai/speech-to-text-translate
SARVAM_MODEL=saaras:v3

# Optional — defaults to "small" if not set
WHISPER_MODEL=small
```

- Get a Groq API key at [console.groq.com](https://console.groq.com/).
- Get a Sarvam AI API key at [sarvam.ai](https://www.sarvam.ai/) (only needed for Hinglish transcription).
- `WHISPER_MODEL` accepts any Faster-Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, etc.) — larger models are more accurate but slower and use more memory.

## Usage

### Option A: Streamlit Web App (recommended)

```bash
streamlit run app.py
```

This opens a browser UI where you can:
1. Choose **YouTube URL** or **Local file path** as the source.
2. Paste the URL or path, and pick a language (**english** or **hinglish**).
3. Click **🚀 Run Pipeline** to process the video/audio.
4. Browse the results across four tabs: **Summary**, **Action Items & Decisions**, **Transcript** (with a download button), and **Chat** (ask questions about the content, answered via RAG).

### Option B: Command Line

```bash
python main.py
```

You'll be prompted to enter:
1. A YouTube URL or local file path.
2. A language (`english` or `hinglish`, defaults to `english` if left blank).

The script prints the title, summary, action items, key decisions, and open questions to the terminal.

## Notes & limitations

- The first run will download the Faster-Whisper model and the HuggingFace embedding model (`all-MiniLM-L6-v2`), which requires an internet connection and may take a few minutes.
- Local file inputs can be any format supported by `pydub`/FFmpeg (e.g. `.mp4`, `.mp3`, `.wav`, `.m4a`).
- The Chroma vector store persists to the local `vector_db/` directory and is rebuilt fresh each time a new transcript is processed.
- Downloaded/converted audio is written to a local `downloads/` folder; both `downloads/` and `vector_db/` are excluded from git via `.gitignore`.

## License

I am the onwer of this repository that is the license.
