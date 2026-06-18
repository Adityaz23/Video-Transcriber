import os

import yt_dlp
from pydub import AudioSegment

# Downloading the audio transcripts in the folder ->

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# This function will download from the youtube url ->
def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "keepvideo": False,
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            },
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    original_file = ydl.prepare_filename(info)

    print("Original file:", original_file)

    filename = os.path.splitext(original_file)[0] + ".wav"

    print("Expected wav:", filename)
    print("Exists?", os.path.exists(filename))

    return filename


# data = download_youtube_audio("https://www.youtube.com/watch?v=_8dJr8nVyDU")


# This function will download the audio from mp4 file, mp3 file or from your
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file into the WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


# data_final = (convert_to_wav(data))


# Now, this function will convert the audio into chunks -> We will mark the
# chunk for 10 mins. Chunking process works in ms so we will convert them to s
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)
    return chunks


# print(chunk_audio(data_final))


# Trigger function ->
def process_input(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected youtube url and downloading audio")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting the file to wav format.")
        wav_path = convert_to_wav(source)
    print("Chunking audio.....")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
