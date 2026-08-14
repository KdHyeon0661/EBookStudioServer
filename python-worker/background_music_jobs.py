import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import numpy as np
import os
import json
import time
import re
import random
import warnings
import hashlib
import importlib.util
from filelock import FileLock, Timeout  # pip install filelock
from typing import Callable
from book_music_metadata import attach_segment_music, save_book_json_atomic
from music_prompt_catalog import (
    DEFAULT_GENERATOR_VERSION,
    MusicPromptCatalog,
    prompt_signature,
)

try:
    from nltk.corpus import wordnet
except ImportError:
    wordnet = None

# indexer 임포트 예외처리
try:
    from indexer import create_music_index
except ImportError:
    create_music_index = None

warnings.filterwarnings('ignore')

# =========================================================
# [Cache & Paths]
# =========================================================
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.dirname(_BASE_DIR)
_STORAGE_ROOT = os.path.abspath(os.environ.get("EBOOK_STORAGE_ROOT", _PROJECT_ROOT))
_DEFAULTS_DIR = os.path.join(_STORAGE_ROOT, "defaults")
_MUSIC_DEFAULTS_DIR = os.path.join(_DEFAULTS_DIR, "music")
_DEFAULT_DB_PATH = os.path.abspath(os.environ.get("EBOOK_DB_PATH", os.path.join(_PROJECT_ROOT, "users.db")))

_KEYWORDS_HISTORY_PATH = os.environ.get("KEYWORDS_HISTORY_PATH") or os.path.join(_DEFAULTS_DIR, "keywords_history.json")
MAX_FILES_PER_FOLDER = 1000


def _normalize_keywords(keywords):
    if not keywords: return []
    out = []
    seen = set()
    for k in keywords:
        if not isinstance(k, str): continue
        k2 = re.sub(r"\s+", " ", k.strip().lower())
        if not k2 or k2 in seen: continue
        seen.add(k2)
        out.append(k2)
    return out[:5]


def _prompt_signature(prompt, genre, bpm, keywords, target_duration_sec, segment_duration):
    return prompt_signature(
        prompt, genre, bpm, keywords, target_duration_sec, segment_duration,
        DEFAULT_GENERATOR_VERSION,
    )


# =========================================================
# [Storage Logic] 공용 폴더 관리 (storage_xxx)
# =========================================================

def find_master_file(filename, music_folder=None):
    root_folder = os.path.abspath(music_folder or _MUSIC_DEFAULTS_DIR)
    if not os.path.exists(root_folder):
        return None
    root_path = os.path.join(root_folder, filename)
    if os.path.exists(root_path):
        return root_path
    for entry in os.scandir(root_folder):
        if entry.is_dir():
            target = os.path.join(entry.path, filename)
            if os.path.exists(target):
                return target
    return None


def get_storage_folder(music_folder=None):
    root_folder = os.path.abspath(music_folder or _MUSIC_DEFAULTS_DIR)
    os.makedirs(root_folder, exist_ok=True)
    subdirs = sorted(
        entry.name for entry in os.scandir(root_folder)
        if entry.is_dir() and entry.name.startswith("storage_")
    )
    if subdirs:
        last_name = subdirs[-1]
        last_path = os.path.join(root_folder, last_name)
        file_count = sum(
            1 for filename in os.listdir(last_path)
            if os.path.isfile(os.path.join(last_path, filename))
        )
        if file_count < MAX_FILES_PER_FOLDER:
            return last_path
        try:
            next_number = int(last_name.split('_')[1]) + 1
        except (IndexError, ValueError):
            next_number = 1
    else:
        next_number = 1
    target = os.path.join(root_folder, f"storage_{next_number:03d}")
    os.makedirs(target, exist_ok=True)
    return target

# ==========================================
# [설정] 감정 -> 장르 매핑
# ==========================================
EMOTION_GENRE_CONNECTOR = {
    "anger": ["metal", "heavy_metal", "industrial", "punk", "drill", "hard_rock", "battle", "epic"],
    "disgust": ["industrial", "noise", "glitch", "experimental", "dark_ambient", "grunge"],
    "fear": ["horror", "dark_ambient", "drone", "suspense", "soundscape", "creepy"],
    "joy": ["pop", "disco", "funk", "house", "edm", "k_pop", "j_pop", "upbeat", "happy_hardcore"],
    "neutral": ["ambient", "minimal", "lofi", "chillout", "easy_listening", "background"],
    "sadness": ["blues", "classical", "piano", "ambient", "ballad", "acoustic", "cello", "noir"],
    "surprise": ["glitch", "idm", "experimental", "jazz_fusion", "dubstep", "progressive_house"]
}


# ==========================================
# [Module Loading]
# ==========================================
def _load_module_from_path(module_name, file_path):
    if not os.path.exists(file_path): return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader: return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except:
        return None


def _defaults_path(filename):
    runtime_path = os.path.join(_DEFAULTS_DIR, filename)
    return runtime_path if os.path.isfile(runtime_path) else os.path.join(_BASE_DIR, "defaults", filename)


MUSIC_GENRES_200 = {}
_gen_mod = _load_module_from_path("defaults_music_genres_200", _defaults_path("music_genres_200.py"))
if _gen_mod and hasattr(_gen_mod, "MUSIC_GENRES_200"):
    MUSIC_GENRES_200 = _gen_mod.MUSIC_GENRES_200

GENRE_BPM_CONNECTOR = {}
_bpm_mod = _load_module_from_path("defaults_genre_bpm_connector", _defaults_path("genre_bpm_connector.py"))
if _bpm_mod and hasattr(_bpm_mod, "GENRE_BPM_CONNECTOR"):
    GENRE_BPM_CONNECTOR = _bpm_mod.GENRE_BPM_CONNECTOR

# ==========================================
# Models
# ==========================================
processor = None
music_model = None


def load_models():
    global processor, music_model
    if music_model is not None and processor is not None:
        return
    print("[MusicGen] Loading...")
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        music_model.to(device)
    except Exception as error:
        processor = None
        music_model = None
        raise RuntimeError(f"MusicGen model could not be loaded: {error}") from error


def crossfade_audio(audio1, audio2, fade_duration_sec, sampling_rate):
    fade_samples = int(fade_duration_sec * sampling_rate)
    if fade_samples == 0: return np.concatenate([audio1, audio2])

    fade_samples = min(fade_samples, len(audio1), len(audio2))
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    overlap = audio1[-fade_samples:] * fade_out + audio2[:fade_samples] * fade_in
    return np.concatenate([audio1[:-fade_samples], overlap, audio2[fade_samples:]])


def get_random_synonym(emotion_key):
    synonyms = {emotion_key}
    if wordnet is not None:
        try:
            for syn in wordnet.synsets(emotion_key):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' ').lower())
        except LookupError:
            pass
    return random.choice(sorted(synonyms))


def estimate_bpm_from_analysis(analysis):
    hints = {"anger": 140, "disgust": 110, "fear": 100, "joy": 125, "neutral": 80, "sadness": 70, "surprise": 130}
    emotion = analysis.get('emotion', 'neutral')
    base_bpm = hints.get(emotion, 90)

    keywords_text = " ".join(analysis.get('keywords', [])).lower()
    if any(w in keywords_text for w in ['fast', 'run', 'rush', 'speed', 'quick', 'urgent']):
        base_bpm += 20
    elif any(w in keywords_text for w in ['slow', 'calm', 'sleep', 'quiet', 'relax', 'peace']):
        base_bpm -= 20

    return int(max(60, min(180, base_bpm)))


def pick_bpm_from_genre_bpm_connector(genre: str, fallback_bpm: int) -> int:
    if not genre: return max(1, int(fallback_bpm))
    rng = GENRE_BPM_CONNECTOR.get(genre)
    if rng is None: return max(1, int(fallback_bpm))

    lo, hi = int(rng[0]), int(rng[1])
    if lo > hi: lo, hi = hi, lo
    center = min(max(int(fallback_bpm), lo), hi)
    return max(1, int(random.triangular(lo, hi, center)))


def select_genre_dynamically(analysis: dict) -> str:
    emotion = analysis.get("emotion", "neutral")
    candidates = []
    if emotion in EMOTION_GENRE_CONNECTOR:
        candidates.extend(EMOTION_GENRE_CONNECTOR[emotion])

    kw_text = " ".join(analysis.get("keywords", []))
    if "space" in kw_text: candidates.append("synthwave")
    if "battle" in kw_text: candidates.append("epic")
    if "love" in kw_text: candidates.append("rnb")

    if not candidates: return "ambient"
    return random.choice(candidates)


def create_dynamic_music_prompt(analysis: dict):
    base_emotion = analysis.get("emotion", "neutral")
    varied_emotion = get_random_synonym(base_emotion)
    keywords = analysis.get("keywords", []) or []
    keywords = keywords[:5]

    selected_genre = select_genre_dynamically(analysis)
    bpm_est = estimate_bpm_from_analysis(analysis)
    bpm = pick_bpm_from_genre_bpm_connector(selected_genre, bpm_est)

    genre_description = ""
    if MUSIC_GENRES_200 and selected_genre in MUSIC_GENRES_200:
        genre_description = MUSIC_GENRES_200[selected_genre]

    prompt_parts = [f"Genre: {selected_genre}"]
    if genre_description:
        prompt_parts.append(f"Style: {genre_description}")
    prompt_parts.append(f"Mood: {varied_emotion}")
    if bpm != 0:
        prompt_parts.append(f"Tempo: {bpm} BPM")
    if keywords:
        prompt_parts.append("Keywords: " + ", ".join(keywords))
    prompt_parts.append("Instrumental background music, cinematic, clean mix, no vocals.")
    prompt = ". ".join(prompt_parts)
    return prompt, selected_genre, int(bpm)


class MusicGenerationCancelled(RuntimeError):
    pass


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise MusicGenerationCancelled("Music generation was cancelled")


def generate_music_segments(prompt, target_duration_sec=120, segment_duration=30,
                            should_cancel: Callable[[], bool] | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if music_model is None or processor is None:
        print("[MusicGen] Model not loaded.")
        return None, None

    sampling_rate = music_model.config.audio_encoder.sampling_rate
    chunk_sec = int(segment_duration)
    overlap_sec = 3
    num_chunks = max(1, int(np.ceil(target_duration_sec / (chunk_sec - overlap_sec))))
    max_tokens = int(chunk_sec * 50)

    final_audio = None
    print(f"      [MusicGen] Generating {target_duration_sec}s for '{prompt[:30]}...' (Total Chunks: {num_chunks})")

    for i in range(num_chunks):
        _raise_if_cancelled(should_cancel)
        try:
            start_time = time.time()
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                audio_values = music_model.generate(
                    **inputs, max_new_tokens=max_tokens, guidance_scale=3.0,
                    do_sample=True, temperature=1.0, top_p=0.95
                )
            chunk_audio = audio_values[0, 0].cpu().float().numpy()

            if final_audio is None:
                final_audio = chunk_audio
            else:
                try:
                    final_audio = crossfade_audio(final_audio, chunk_audio, overlap_sec, sampling_rate)
                except:
                    final_audio = np.concatenate([final_audio, chunk_audio])

            del inputs, audio_values
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            elapsed = time.time() - start_time
            print(f"         [MusicGen] Chunk {i + 1}/{num_chunks} Generated ({elapsed:.1f}s)")
            _raise_if_cancelled(should_cancel)
            time.sleep(0.5)
        except MusicGenerationCancelled:
            raise
        except Exception as e:
            print(f"         [MusicGen] Chunk {i + 1} fail: {e}")
            continue

    if final_audio is not None and len(final_audio) > 0:
        max_val = np.max(np.abs(final_audio))
        if max_val > 0: final_audio = final_audio / max_val * 0.9
        return final_audio, sampling_rate
    else:
        return None, None


def _deterministic_prompt(analysis, identity):
    state = random.getstate()
    try:
        seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
        random.seed(seed)
        return create_dynamic_music_prompt(analysis)
    finally:
        random.setstate(state)



def _segment_content_signature(segment):
    hint = segment.get("generation_hint") or {}
    supplied = str(hint.get("content_signature") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", supplied):
        return supplied
    text = " ".join(str(page.get("text") or "") for page in segment.get("pages") or [])
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def process_book_background(json_path, music_folder, web_path_prefix, username=None, book_id=None,
                            should_cancel: Callable[[], bool] | None = None,
                            catalog_db_path=None, job_id=None):
    """Generate missing segment music and atomically attach it to the current book JSON."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Book JSON not found: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as source:
            data = json.load(source)
    except Exception as error:
        raise RuntimeError(f"Book JSON could not be loaded: {error}") from error

    chapters = data.get("chapters") or []
    if not chapters:
        raise RuntimeError("Book JSON has no chapters")

    requested = generated = reused = 0
    failures = []
    changed = False
    root_folder = os.path.abspath(music_folder or _MUSIC_DEFAULTS_DIR)
    os.makedirs(root_folder, exist_ok=True)
    catalog = MusicPromptCatalog(catalog_db_path or _DEFAULT_DB_PATH, root_folder)
    lock_folder = os.path.join(root_folder, ".locks")
    os.makedirs(lock_folder, exist_ok=True)
    print(f"[Gen] Start background generation for book: {book_id}")

    for chapter_index, chapter in enumerate(chapters):
        for segment_index, segment in enumerate(chapter.get("segments") or []):
            _raise_if_cancelled(should_cancel)
            hint = segment.get("generation_hint")
            if not hint:
                continue
            requested += 1
            try:
                target_emotion = hint.get("target_emotion", "neutral")
                keywords = _normalize_keywords(hint.get("keywords", []))
                analysis = {"emotion": target_emotion, "keywords": keywords}
                content_id = _segment_content_signature(segment)
                identity = f"{content_id}:{target_emotion}:{','.join(keywords)}"
                prompt, genre, bpm = _deterministic_prompt(analysis, identity)
                target_duration = 120
                segment_duration = 30
                signature = _prompt_signature(prompt, genre, int(bpm), keywords, target_duration, segment_duration)
                filename = f"{signature}.wav"
                metadata = {
                    "prompt": prompt, "genre": genre, "bpm": int(bpm), "keywords": keywords,
                    "target_duration_sec": target_duration,
                    "segment_duration_sec": segment_duration,
                    "generator_version": DEFAULT_GENERATOR_VERSION,
                }
                master_path = catalog.find_ready(signature)
                if master_path is None:
                    master_path = find_master_file(filename, root_folder)
                    if master_path is not None:
                        catalog.mark_ready(
                            signature, metadata, filename, master_path, job_id, reused=True
                        )
                source_type = "ai_reused"

                if master_path is None:
                    lock = FileLock(os.path.join(lock_folder, f"{signature}.lock"))
                    deadline = time.monotonic() + 300
                    while True:
                        _raise_if_cancelled(should_cancel)
                        try:
                            lock.acquire(timeout=1)
                            break
                        except Timeout:
                            if time.monotonic() >= deadline:
                                raise RuntimeError("Timed out waiting for identical prompt generation")
                    try:
                        master_path = catalog.find_ready(signature)
                        if master_path is None:
                            master_path = find_master_file(filename, root_folder)
                        if master_path is not None:
                            catalog.mark_ready(
                                signature, metadata, filename, master_path, job_id, reused=True
                            )
                            reused += 1
                        else:
                            catalog.mark_generating(signature, metadata, filename, job_id)
                            try:
                                load_models()
                                audio, sample_rate = generate_music_segments(
                                    prompt, target_duration, segment_duration,
                                    should_cancel=should_cancel,
                                )
                                if audio is None or sample_rate is None:
                                    raise RuntimeError("MusicGen returned no audio")
                                save_dir = get_storage_folder(root_folder)
                                target_path = os.path.join(save_dir, filename)
                                temp_path = target_path + ".tmp"
                                audio = np.clip(audio, -1.0, 1.0)
                                wavfile.write(temp_path, sample_rate, (audio * 32767).astype(np.int16))
                                os.replace(temp_path, target_path)
                                metadata_path = target_path.replace(".wav", ".json")
                                with open(metadata_path + ".tmp", "w", encoding="utf-8") as output:
                                    json.dump({
                                        "signature": signature,
                                        "generator_version": DEFAULT_GENERATOR_VERSION,
                                        "prompt": prompt, "emotion": target_emotion,
                                        "genre": genre, "bpm": bpm, "keywords": keywords,
                                        "created_at": time.time(),
                                    }, output, ensure_ascii=False, indent=2)
                                os.replace(metadata_path + ".tmp", metadata_path)
                                master_path = target_path
                                catalog.mark_ready(
                                    signature, metadata, filename, master_path, job_id
                                )
                                generated += 1
                                source_type = "ai_generated"
                            except MusicGenerationCancelled as error:
                                catalog.mark_failed(signature, str(error), cancelled=True)
                                raise
                            except Exception as error:
                                catalog.mark_failed(signature, str(error))
                                raise
                    finally:
                        lock.release()
                else:
                    reused += 1

                _raise_if_cancelled(should_cancel)
                attach_segment_music(data, chapter_index, segment_index, filename, bpm, source_type)
                changed = True
            except MusicGenerationCancelled:
                raise
            except Exception as error:
                failures.append(f"chapter {chapter_index + 1}, segment {segment_index + 1}: {error}")

    _raise_if_cancelled(should_cancel)
    if changed:
        save_book_json_atomic(json_path, data)
    if failures:
        raise RuntimeError("; ".join(failures[:10]))
    print(f"[Gen] Finished for {book_id}: requested={requested}, generated={generated}, reused={reused}")
    return {"requested": requested, "generated": generated, "reused": reused}
