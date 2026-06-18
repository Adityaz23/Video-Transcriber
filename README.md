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

`requirements.txt` is aligned with what the code actually imports: **Faster-Whisper** for local transcription and **Groq** (`langchain-groq`) for all LLM calls (summarization, extraction, RAG).

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

## Deploying for free (Streamlit Community Cloud)

This repo is set up to deploy on [Streamlit Community Cloud](https://share.streamlit.io) at no cost:

1. Push this repo (including `packages.txt`, which installs FFmpeg) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Select this repository, branch `main`, and set the main file path to `app.py`.
4. Under **Advanced settings → Secrets**, add your API keys:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   SARVAM_API_KEY = "your_sarvam_api_key"
   SARVAM_SST_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
   ```
5. Click **Deploy**.

Your app will be live at a public URL like `https://<your-app-name>.streamlit.app`.

On a hosted deployment, only the **YouTube URL** input is shown by default (the "Local file path" option only makes sense when running on your own machine, since the server's filesystem isn't the visitor's). To re-enable local file input for local development, set `ALLOW_LOCAL_FILE_INPUT=true` in your `.env` file.

**Free-tier limits to be aware of:** the app sleeps after ~15 minutes of inactivity (taking 30–60 seconds to wake on the next visit), and CPU/RAM is limited — stick to the `small` or smaller Faster-Whisper model (`WHISPER_MODEL=small` or `tiny`) to avoid running out of memory on longer videos.

**Known issue — YouTube downloads on hosted/cloud deployments:** YouTube actively rate-limits and blocks downloads from datacenter IPs (the kind cloud hosts like Streamlit Cloud use), which can surface as `HTTP Error 403: Forbidden` even though the same URL works fine when run locally. This is a YouTube-side anti-bot measure, not a bug in this app, and there's no fully reliable free fix — `yt-dlp` and YouTube are in an ongoing cat-and-mouse cycle. Keeping `yt-dlp` updated to its latest version helps somewhat. Running the app locally (`streamlit run app.py` on your own machine) avoids the issue entirely, since your home IP isn't flagged the way cloud IPs are.

## License

No license file is currently included in this repository.
