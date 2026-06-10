import os
import requests

from faster_whisper import WhisperModel

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_TRANSLATE_URL = os.getenv("SARVAM_SST_TRANSLATE_URL")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v3")


def load_model():
    global _model
    if _model is None:
        print(f"Loading Faster-Whisper model '{WHISPER_MODEL}'...")
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("Model loaded successfully")
    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk not found: {chunk_path}")
    model = load_model()
    task = "translate" if translate else "transcribe"
    segments, _ = model.transcribe(chunk_path, language="hi", task=task)
    return " ".join(segment.text for segment in segments)


def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        try:
            text = transcribe_chunk(chunk, translate=translate)
            full_transcript += text + " "
        except Exception as e:
            print(f"Failed on chunk {i + 1}: {e}")
    print("Transcription completed")
    return full_transcript.strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError(f"No sarvam api key found")
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path, f, "audio/wav"))}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_TRANSLATE_URL, headers=headers, files=files, data=data, timeout=3000
        )
    response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_with_sarvam(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper to Sarvam depending on language choice.
    - english -> Whisper(local model)
    - hinglish -> Sarvam(translates to English while trancribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk(chunk_path)


def transcibe_all_chunk(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    engine = "SARVAM AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using{engine} for transcript")

    for i in chunks in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}....")
        text = transcribe_chunk(chunks, language=language)
        full_transcript += text + " "
    print(f"Transcript complete")
    return full_transcript.strip()
