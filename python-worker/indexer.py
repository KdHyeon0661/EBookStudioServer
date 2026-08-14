import hashlib
import json
import os
import warnings

import librosa
import numpy as np
from filelock import FileLock, Timeout

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STORAGE_ROOT = os.path.abspath(os.environ.get("EBOOK_STORAGE_ROOT", PROJECT_ROOT))
DEFAULT_MUSIC_FOLDER = os.path.join(STORAGE_ROOT, "defaults", "music")


def analyze_audio(file_path):
    try:
        try:
            duration = librosa.get_duration(path=file_path)
        except Exception:
            samples, sample_rate = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=samples, sr=sample_rate)
        samples, sample_rate = librosa.load(file_path, sr=None, duration=60)
        tempo, _ = librosa.beat.beat_track(y=samples, sr=sample_rate)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0] if len(tempo) else 0.0
        return duration, float(tempo)
    except Exception as error:
        print(f"[Indexer] Analysis failed ({os.path.basename(file_path)}): {error}")
        return 0.0, 0.0


def create_music_index(music_folder=None):
    folder = os.path.abspath(music_folder or DEFAULT_MUSIC_FOLDER)
    defaults_folder = os.path.dirname(folder)
    index_file = os.path.join(defaults_folder, "music_index.json")
    lock_file = index_file + ".lock"
    os.makedirs(folder, exist_ok=True)
    try:
        with FileLock(lock_file, timeout=60):
            return _build_index(folder, index_file)
    except Timeout as error:
        raise RuntimeError("Music index is locked by another worker") from error


def _build_index(music_folder, index_file):
    valid_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    disk_files = {}
    for root, _, files in os.walk(music_folder):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                disk_files[filename] = os.path.join(root, filename)

    index_data = {}
    if os.path.isfile(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as source:
                index_data = json.load(source)
        except Exception:
            index_data = {}

    changed = False
    for key in list(index_data):
        if index_data[key].get("filename") not in disk_files:
            del index_data[key]
            changed = True

    registered = {value.get("filename") for value in index_data.values()}
    for filename in sorted(set(disk_files) - registered):
        file_path = disk_files[filename]
        metadata_path = os.path.splitext(file_path)[0] + ".json"
        genre, prompt, bpm = "unknown", "", 0
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                genre = metadata.get("genre") or metadata.get("emotion") or genre
                prompt = metadata.get("original_prompt") or metadata.get("prompt") or prompt
                bpm = metadata.get("bpm", 0)
            except Exception:
                pass
        duration, detected_bpm = analyze_audio(file_path)
        if not bpm:
            bpm = detected_bpm
        file_id = int(hashlib.sha256(filename.encode("utf-8")).hexdigest()[:14], 16)
        while str(file_id) in index_data and index_data[str(file_id)].get("filename") != filename:
            file_id += 1
        index_data[str(file_id)] = {
            "id": file_id,
            "genre": genre,
            "filename": filename,
            "duration": round(duration, 2),
            "bpm": int(round(bpm)),
            "prompt": prompt or f"{genre} mood music",
            "is_custom": filename != "default_ambient.wav",
        }
        changed = True

    if changed or not os.path.isfile(index_file):
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        temp_file = index_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as output:
            json.dump(index_data, output, indent=2, ensure_ascii=False)
        os.replace(temp_file, index_file)
    print(f"[Indexer] Indexed {len(index_data)} music files from {music_folder}")
    return {"count": len(index_data), "index_file": index_file}


if __name__ == "__main__":
    create_music_index()