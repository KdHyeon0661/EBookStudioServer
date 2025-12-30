import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline
from keybert import KeyBERT
import scipy.io.wavfile as wavfile
import numpy as np
import os
import json
import time
import re
import random
import warnings
import threading
import uuid
import hashlib
import importlib.util

import nltk
from nltk.corpus import wordnet

# indexer 임포트 예외처리
try:
    from indexer import create_music_index
except ImportError:
    create_music_index = None

warnings.filterwarnings('ignore')

# NLTK 다운로드
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("[NLTK] WordNet Downloading...")
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"⚠️ NLTK Download Failed: {e}")

# =========================================================
# [Cache & Paths]
# =========================================================
_KEYWORDS_HISTORY_LOCK = threading.Lock()
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
_DEFAULTS_DIR = os.path.join(_BASE_DIR, "defaults")
_MUSIC_DEFAULTS_DIR = os.path.join(_DEFAULTS_DIR, "music")

_KEYWORDS_HISTORY_PATH = os.environ.get("KEYWORDS_HISTORY_PATH") or os.path.join(_DEFAULTS_DIR, "keywords_history.json")

# [설정] 스토리지 폴더 당 최대 파일 개수
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


def _load_keywords_history() -> dict:
    try:
        if os.path.exists(_KEYWORDS_HISTORY_PATH):
            with open(_KEYWORDS_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        return {}
    return {}


def _save_keywords_history(hist: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_KEYWORDS_HISTORY_PATH), exist_ok=True)
        tmp = _KEYWORDS_HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _KEYWORDS_HISTORY_PATH)
    except:
        pass


def _prompt_signature(prompt, genre, bpm, keywords, target_duration_sec, segment_duration):
    payload = {
        "prompt": prompt, "genre": genre, "bpm": bpm, "keywords": keywords,
        "target_duration_sec": target_duration_sec, "segment_duration": segment_duration,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


# =========================================================
# [Storage Logic] 공용 폴더 관리 (storage_xxx)
# =========================================================

def find_master_file(filename):
    if not os.path.exists(_MUSIC_DEFAULTS_DIR):
        return None
    root_path = os.path.join(_MUSIC_DEFAULTS_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    for entry in os.scandir(_MUSIC_DEFAULTS_DIR):
        if entry.is_dir():
            target = os.path.join(entry.path, filename)
            if os.path.exists(target):
                return target
    return None


def get_storage_folder():
    if not os.path.exists(_MUSIC_DEFAULTS_DIR):
        os.makedirs(_MUSIC_DEFAULTS_DIR, exist_ok=True)
    subdirs = []
    for entry in os.scandir(_MUSIC_DEFAULTS_DIR):
        if entry.is_dir() and entry.name.startswith("storage_"):
            subdirs.append(entry.name)
    subdirs.sort()

    target_dir = None
    if subdirs:
        last_dir_name = subdirs[-1]
        last_dir_path = os.path.join(_MUSIC_DEFAULTS_DIR, last_dir_name)
        file_count = len([f for f in os.listdir(last_dir_path) if os.path.isfile(os.path.join(last_dir_path, f))])
        if file_count < MAX_FILES_PER_FOLDER:
            target_dir = last_dir_path
        else:
            try:
                last_num = int(last_dir_name.split('_')[1])
                new_name = f"storage_{last_num + 1:03d}"
                target_dir = os.path.join(_MUSIC_DEFAULTS_DIR, new_name)
            except:
                target_dir = os.path.join(_MUSIC_DEFAULTS_DIR, "storage_001")
    else:
        target_dir = os.path.join(_MUSIC_DEFAULTS_DIR, "storage_001")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    return target_dir


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


def _defaults_path(filename): return os.path.join(_DEFAULTS_DIR, filename)


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
kw_model = None
emotion_classifier = None


def load_models():
    global processor, music_model, kw_model, emotion_classifier

    if music_model is None:
        print("[MusicGen] Loading...")
        try:
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            music_model.to(device)
        except Exception as e:
            print(f"❌ MusicGen Error: {e}")

    if kw_model is None:
        print("[KeyBERT] Loading...")
        try:
            kw_model = KeyBERT('roberta-base')
        except Exception as e:
            print(f"❌ KeyBERT Error: {e}")

    if emotion_classifier is None:
        print("[RoBERTa] Loading...")
        try:
            emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"❌ Emotion Classifier Error: {e}")


def analyze_text_with_keybert_roberta(text):
    target_text = text[:512]
    base_emotion = 'neutral'
    if emotion_classifier:
        try:
            emotion_result = emotion_classifier(target_text)[0]
            base_emotion = emotion_result['label'].lower()
        except:
            pass

    top_keywords = []
    if kw_model:
        try:
            keywords = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5,
                diversity=0.7, use_mmr=True
            )
            top_keywords = [kw[0] for kw in keywords]
        except:
            pass

    return {'emotion': base_emotion, 'keywords': top_keywords, 'original_emotion': base_emotion}


def crossfade_audio(audio1, audio2, fade_duration_sec, sampling_rate):
    fade_samples = int(fade_duration_sec * sampling_rate)
    if fade_samples == 0: return np.concatenate([audio1, audio2])

    fade_samples = min(fade_samples, len(audio1), len(audio2))
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    overlap = audio1[-fade_samples:] * fade_out + audio2[:fade_samples] * fade_in
    return np.concatenate([audio1[:-fade_samples], overlap, audio2[fade_samples:]])


def get_random_synonym(emotion_key):
    synonyms = set([emotion_key])
    try:
        for syn in wordnet.synsets(emotion_key):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' ').lower())
    except:
        pass
    return random.choice(list(synonyms))


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


def generate_music_segments(prompt, target_duration_sec=120, segment_duration=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if music_model is None or processor is None:
        print("❌ Model not loaded.")
        return None, None

    sampling_rate = music_model.config.audio_encoder.sampling_rate
    chunk_sec = int(segment_duration)
    overlap_sec = 3
    num_chunks = max(1, int(np.ceil(target_duration_sec / (chunk_sec - overlap_sec))))
    max_tokens = int(chunk_sec * 50)

    final_audio = None
    print(f"      [MusicGen] Generating {target_duration_sec}s for '{prompt[:30]}...' (Total Chunks: {num_chunks})")

    for i in range(num_chunks):
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
            print(f"         ⏳ Chunk {i + 1}/{num_chunks} 생성 완료 ({elapsed:.1f}초 소요)")
            time.sleep(0.5)
        except Exception as e:
            print(f"         ❌ Chunk {i + 1} fail: {e}")
            continue

    if final_audio is not None and len(final_audio) > 0:
        max_val = np.max(np.abs(final_audio))
        if max_val > 0: final_audio = final_audio / max_val * 0.9
        return final_audio, sampling_rate
    else:
        return None, None


def process_book_background(json_path, music_folder, web_path_prefix, username=None, book_id=None):
    load_models()

    if os.path.isdir(json_path):
        paths = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith(".json")]
    else:
        paths = [json_path]

    for path in paths:
        if not os.path.exists(path): continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load JSON {path}: {e}")
            continue

        updated = False
        chapters = data.get("chapters", [])

        if not chapters and isinstance(data.get("segments"), list):
            chapters = [{"chapter_index": 0, "segments": data.get("segments")}]

        if not chapters: continue

        for ci, chapter in enumerate(chapters):
            segments = chapter.get("segments", []) or []

            for si, segment in enumerate(segments):
                try:
                    current_name = str(segment.get("music_filename", "") or "")

                    # [중요] 기존 파일이 있어도 무시하고 생성하도록 해당 체크 로직 제거됨.

                    pages = segment.get("pages", []) or []
                    texts = [p.get("text", "").strip() for p in pages if isinstance(p, dict)]
                    combined_text = " ".join(texts).strip()
                    if not combined_text: continue

                    print(f"🔍 [Analyzing] Ch{ci}-Seg{si} ({len(combined_text)} chars)")

                    analysis = analyze_text_with_keybert_roberta(combined_text)
                    prompt, genre, bpm = create_dynamic_music_prompt(analysis)

                    print(f"   👉 Mood: {analysis.get('emotion')} | Genre: {genre} | BPM: {bpm}")

                    target_duration = 120
                    seg_dur = 30
                    norm_keywords = _normalize_keywords(analysis.get("keywords", []))

                    sig = _prompt_signature(prompt, genre, int(bpm), norm_keywords, target_duration, seg_dur)
                    filename = f"{sig}.wav"

                    master_path = find_master_file(filename)
                    music_source = "ai_gen"

                    if master_path:
                        print(f"♻️ [Reuse] Found in storage: {master_path}")
                        music_source = "ai_reused"
                    else:
                        print(f"🎹 [New] Generating: {filename}")
                        audio, sr = generate_music_segments(prompt, target_duration, seg_dur)

                        if audio is not None and sr is not None:
                            save_dir = get_storage_folder()
                            master_path = os.path.join(save_dir, filename)

                            audio = np.clip(audio, -1.0, 1.0)
                            audio_i16 = (audio * 32767).astype(np.int16)
                            wavfile.write(master_path, sr, audio_i16)

                            meta_path = master_path.replace(".wav", ".json")
                            with open(meta_path, "w", encoding="utf-8") as mf:
                                json.dump({
                                    "prompt": prompt, "emotion": analysis.get("emotion"),
                                    "genre": genre, "bpm": bpm, "keywords": norm_keywords
                                }, mf, ensure_ascii=False, indent=2)

                            print(f"💾 Saved to Storage: {master_path}")
                        else:
                            print("❌ Audio generation failed.")
                            continue

                    segment["music_filename"] = filename
                    segment["music_path"] = f"music/{filename}"
                    segment["music_source"] = music_source
                    segment["emotion"] = analysis.get("emotion", "neutral")
                    segment["genre"] = genre
                    if bpm != 0: segment["bpm"] = bpm

                    updated = True
                    time.sleep(0.5)

                except Exception as e:
                    print(f"[BG] Error in Segment {si}: {e}")
                    continue

        if updated:
            try:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)
                print(f"✅ JSON Updated: {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Failed to update JSON file: {e}")


class BackgroundMusicJobRunner:
    def __init__(self, users_folder, queue_file=None):
        self.users_folder = users_folder
        self.queue_file = queue_file or os.path.join(users_folder, "_bg_jobs.json")
        self._lock = threading.Lock()
        self._jobs = []
        self._load()

        # [추가됨] 서버 켜질 때 'running' 상태인 작업 복구 (Zombie Jobs Recovery)
        self._recover_stuck_jobs()

    def _load(self):
        try:
            if os.path.exists(self.queue_file):
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    self._jobs = json.load(f)
            else:
                self._jobs = []
        except:
            self._jobs = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)
            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump(self._jobs, f, ensure_ascii=False, indent=2)
        except:
            pass

    # [추가됨] 중단된 작업 복구 메서드
    def _recover_stuck_jobs(self):
        with self._lock:
            self._load()
            recovered_count = 0
            for job in self._jobs:
                # 상태가 'running'인 채로 멈춰있는 녀석들을 찾음
                if job.get("status") == "running":
                    print(f"♻️ [Recovery] 중단된 작업 복구: {job.get('book_id')} (ID: {job.get('id')})")
                    job["status"] = "queued"  # 다시 대기열로
                    job["started_at"] = 0
                    recovered_count += 1

            if recovered_count > 0:
                self._save()
                print(f"✅ 총 {recovered_count}개의 중단된 작업을 대기열로 복구했습니다.")

    def enqueue(self, job_type, username, book_id, json_path=None, music_folder=None,
                web_path_prefix=None, pdf_path=None, book_root_folder=None):
        with self._lock:
            self._load()
            job_id = str(uuid.uuid4())
            new_job = {
                "id": job_id,
                "type": job_type,
                "username": username,
                "book_id": book_id,
                "status": "queued",
                "created_at": int(time.time()),
                "json_path": json_path,
                "music_folder": music_folder,
                "web_path_prefix": web_path_prefix,
                "pdf_path": pdf_path,
                "book_root_folder": book_root_folder
            }
            self._jobs.append(new_job)
            self._save()
            return job_id

    def execute(self, max_jobs=1):
        to_run = []
        with self._lock:
            self._load()
            for j in self._jobs:
                if j.get("status") == "queued":
                    j["status"] = "running"
                    j["started_at"] = int(time.time())
                    to_run.append(j)
                    if len(to_run) >= max_jobs: break
            self._save()

        ran = 0
        for job in to_run:
            ran += 1
            status = "error"
            err = None
            try:
                if job["type"] == "analyze" or job["type"] == "analyze_and_music":
                    print(f"📘 [Job] Analyzing Book: {job.get('book_id', 'Unknown')}", flush=True)
                    try:
                        from analyzer import process_full_book_for_offline
                    except ImportError:
                        raise ImportError("analyzer.py module not found.")

                    result = process_full_book_for_offline(
                        pdf_path=job["pdf_path"],
                        book_root_folder=job["book_root_folder"],
                        music_folder=job["music_folder"],
                        web_path_prefix=job["web_path_prefix"]
                    )

                    # [디버깅] 분석기 결과 출력
                    print(f"   🔍 [Debug] 분석기 반환값: {result}", flush=True)

                    if result and 'text_file' in result:
                        full_json_path = os.path.join(job["book_root_folder"], result['text_file'])

                        print(f"   ↪️ 분석 완료. 음악 생성 작업(Music)을 큐에 추가합니다.", flush=True)
                        self.enqueue(
                            job_type='music',
                            username=job["username"],
                            book_id=job["book_id"],
                            json_path=full_json_path,
                            music_folder=job["music_folder"],
                            web_path_prefix=job["web_path_prefix"]
                        )
                    status = "done"

                elif job["type"] == "music":
                    print(f"🎹 [Job] Generating Music: {job.get('book_id', 'Unknown')}", flush=True)

                    process_book_background(
                        job["json_path"],
                        job["music_folder"],
                        job["web_path_prefix"],
                        job.get("username"),
                        job.get("book_id")
                    )

                    if create_music_index:
                        try:
                            create_music_index()
                        except:
                            pass

                    print(f"🎹 [Job] Generating Music Finished!", flush=True)
                    status = "done"
                else:
                    print(f"⚠️ [Job] 알 수 없는 작업 타입: {job['type']}", flush=True)
                    status = "skipped"

            except Exception as e:
                err = str(e)
                print(f"❌ Job Failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

            with self._lock:
                self._load()
                for j in self._jobs:
                    if j.get("id") == job.get("id"):
                        j["status"] = status
                        j["finished_at"] = int(time.time())
                        j["error"] = err
                        break

                # [청소] 완료된 작업(done, skipped)은 리스트에서 삭제
                self._jobs = [j for j in self._jobs if j["status"] not in ["done", "skipped"]]
                self._save()

        return {"ran": ran}