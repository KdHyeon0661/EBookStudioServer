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
import shutil
from filelock import FileLock  # pip install filelock

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
        print(f"[NLTK] Download Failed: {e}")

# =========================================================
# [Cache & Paths]
# =========================================================
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
_DEFAULTS_DIR = os.path.join(_BASE_DIR, "defaults")
_MUSIC_DEFAULTS_DIR = os.path.join(_DEFAULTS_DIR, "music")

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
            print(f"[MusicGen] Error: {e}")

    if kw_model is None:
        print("[KeyBERT] Loading...")
        try:
            kw_model = KeyBERT('roberta-base')
        except Exception as e:
            print(f"[KeyBERT] Error: {e}")

    if emotion_classifier is None:
        print("[RoBERTa] Loading...")
        try:
            emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"[Emotion Classifier] Error: {e}")


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
            time.sleep(0.5)
        except Exception as e:
            print(f"         [MusicGen] Chunk {i + 1} fail: {e}")
            continue

    if final_audio is not None and len(final_audio) > 0:
        max_val = np.max(np.abs(final_audio))
        if max_val > 0: final_audio = final_audio / max_val * 0.9
        return final_audio, sampling_rate
    else:
        return None, None


def process_book_background(json_path, music_folder, web_path_prefix, username=None, book_id=None):
    """
    [GPU Worker Only]
    이미 분석이 완료된 JSON 파일을 읽어서,
    'generation_hint'가 있는 세그먼트에 대해 실제로 새로운 AI 음악을 생성합니다.
    생성된 음악은 공용 Storage에 저장되어 라이브러리를 확장합니다.
    """
    load_models()  # GPU 모델 로드 (MusicGen)

    if not os.path.exists(json_path):
        print(f"[Gen] JSON not found: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Gen] Failed to load JSON: {e}")
        return

    chapters = data.get("chapters", [])
    if not chapters: return

    print(f"[Gen] Start background generation for book: {book_id}")

    # JSON의 모든 챕터/세그먼트를 순회하며 생성 작업 수행
    for ci, chapter in enumerate(chapters):
        segments = chapter.get("segments", []) or []

        for si, segment in enumerate(segments):
            try:
                # 1. 힌트 확인 (Analyzer가 남겨둔 메모)
                hint = segment.get("generation_hint")
                if not hint:
                    continue  # 힌트가 없으면 생성할 필요 없음 (이미 완벽 매칭 등)

                # 2. 생성 파라미터 추출
                # analyzer가 분석해둔 감정과 키워드를 그대로 사용 (다시 분석 X)
                target_emotion = hint.get("target_emotion", "neutral")
                keywords = hint.get("keywords", [])

                # 장르와 BPM은 랜덤성을 위해 여기서 다시 살짝 변주를 줄 수도 있고,
                # 일관성을 위해 고정할 수도 있음. 여기서는 '다양성'을 위해 랜덤 선택 로직 활용.
                # (이미 analyzer.py에도 있는 로직이지만, 독립적으로 수행)
                fake_analysis_for_gen = {"emotion": target_emotion, "keywords": keywords}
                prompt, genre, bpm = create_dynamic_music_prompt(fake_analysis_for_gen)

                target_duration = 120
                seg_dur = 30
                norm_keywords = _normalize_keywords(keywords)

                # 3. 파일 시그니처(해시) 생성
                sig = _prompt_signature(prompt, genre, int(bpm), norm_keywords, target_duration, seg_dur)
                filename = f"{sig}.wav"

                # 4. 중복 체크 (이미 누가 만들었으면 스킵)
                master_path = find_master_file(filename)
                if master_path:
                    print(f"[Gen] Skip (Already exists): {filename}")
                    continue

                # 5. [GPU Heavy Task] 실제 음악 생성
                # 파일 락을 걸어서 동시에 같은 음악을 만드는 것을 방지
                save_dir = get_storage_folder()
                lock_path = os.path.join(save_dir, f".{filename}.lock")

                print(f"[Gen] Generating new music: {filename} (Genre: {genre})")

                # FileLock으로 안전하게 생성
                with FileLock(lock_path, timeout=300):
                    # 락 얻은 후 다시 확인 (Double Check)
                    if find_master_file(filename):
                        continue

                    audio, sr = generate_music_segments(prompt, target_duration, seg_dur)

                    if audio is not None and sr is not None:
                        target_path = os.path.join(save_dir, filename)
                        temp_path = target_path + ".tmp"

                        # Wav 저장
                        audio = np.clip(audio, -1.0, 1.0)
                        audio_i16 = (audio * 32767).astype(np.int16)
                        wavfile.write(temp_path, sr, audio_i16)
                        os.replace(temp_path, target_path)

                        # 메타데이터 저장 (Indexer가 나중에 읽음)
                        meta_path = target_path.replace(".wav", ".json")
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump({
                                "prompt": prompt,
                                "emotion": target_emotion,
                                "genre": genre,
                                "bpm": bpm,
                                "keywords": norm_keywords,
                                "created_at": time.time()
                            }, mf, ensure_ascii=False, indent=2)

                        print(f"[Gen] Saved new asset: {target_path}")
                    else:
                        print("[Gen] Failed to generate audio.")

                # GPU 과열 방지 및 다른 프로세스 양보를 위한 짧은 휴식
                time.sleep(1.0)

            except Exception as e:
                print(f"[Gen] Error in Segment {si}: {e}")
                continue

    print(f"[Gen] Background generation finished for {book_id}")


# =========================================================
# [Job Runner with DB Support]
# =========================================================
class BackgroundMusicJobRunner:
    def __init__(self, users_folder, db_instance, job_model):
        self.users_folder = users_folder
        self.db = db_instance
        self.JobModel = job_model

    def recover_stuck_jobs(self):
        """
        서버 재시작 시 'running' 상태인 작업을 'queued'로 복구
        """
        try:
            stuck_jobs = self.JobModel.query.filter_by(status='running').all()
            if stuck_jobs:
                print(f"[Recovery] Recovering {len(stuck_jobs)} stuck jobs...")
                for job in stuck_jobs:
                    job.status = 'queued'
                    job.started_at = None
                self.db.session.commit()
                print(f"[Recovery] Done.")
        except Exception as e:
            print(f"[Recovery] Failed: {e}")
            self.db.session.rollback()

    def enqueue(self, job_type, user_uuid, book_id, json_path=None, music_folder=None,
                web_path_prefix=None, pdf_path=None, book_root_folder=None):
        try:
            job_id = str(uuid.uuid4())
            new_job = self.JobModel(
                id=job_id,
                type=job_type,
                user_uuid=user_uuid,
                book_id=book_id,
                status="queued",
                json_path=json_path,
                music_folder=music_folder,
                web_path_prefix=web_path_prefix,
                pdf_path=pdf_path,
                book_root_folder=book_root_folder
            )
            self.db.session.add(new_job)
            self.db.session.commit()
            return job_id
        except Exception as e:
            self.db.session.rollback()
            print(f"[Enqueue] Error: {e}")
            return None

    def execute(self, job_types=None, max_jobs=1):
        """
        DB에서 'queued' 작업을 가져와 실행.
        job_types: 실행할 작업 타입 리스트 (예: ['analyze'] 또는 ['music_generation'])
        - None일 경우 안전을 위해 실행하지 않거나 기본값 처리
        """
        processed_count = 0

        # 워커 프로세스(CPU/GPU)에 따라 처리할 작업 타입을 필터링
        if job_types is None:
            target_types = ['analyze', 'music_generation']
        else:
            target_types = job_types

        # 최대 max_jobs 만큼 반복
        for _ in range(max_jobs):
            target_job = None

            try:
                # 1. 'queued' 작업 하나 찾기 (요청된 타입 중에서, 오래된 순)
                candidate = self.JobModel.query.filter(
                    self.JobModel.status == 'queued',
                    self.JobModel.type.in_(target_types)
                ).order_by(self.JobModel.created_at.asc()).first()

                if not candidate:
                    break  # 대기 중인 작업 없음

                # 2. [Critical Section] 상태를 'running'으로 원자적 업데이트
                rows_updated = self.JobModel.query.filter_by(id=candidate.id, status='queued').update({
                    'status': 'running',
                    'started_at': int(time.time())
                })
                self.db.session.commit()

                if rows_updated == 0:
                    # 다른 프로세스가 이미 가져감 -> 다음 루프
                    continue

                # 업데이트 성공, 작업 획득
                target_job = self.JobModel.query.get(candidate.id)

            except Exception as e:
                self.db.session.rollback()
                print(f"[DB] Selection Error: {e}")
                break

            if not target_job:
                continue

            # 3. 작업 실행
            processed_count += 1
            status = "error"
            err_msg = None

            try:
                # =========================================================
                # [Case A] 분석 작업 (CPU Worker 담당)
                # 목적: PDF -> JSON 변환 및 '기존 음악' 매핑 (사용자 서비스용)
                # =========================================================
                if target_job.type == "analyze":
                    print(f"[Job-CPU] Analyzing Book: {target_job.book_id}")
                    try:
                        from analyzer import process_full_book_for_offline

                        # (1) 분석 수행 (여기서 analyzer.py가 기존 음악을 찾아 매핑함)
                        result = process_full_book_for_offline(
                            pdf_path=target_job.pdf_path,
                            book_root_folder=target_job.book_root_folder,
                            music_folder=target_job.music_folder,
                            web_path_prefix=target_job.web_path_prefix
                        )

                        if result and 'text_file' in result:
                            full_json_path = os.path.join(target_job.book_root_folder, result['text_file'])

                            # (2) [핵심] 분석 완료 후, 라이브러리 확장을 위한 '음악 생성' 작업을 별도 큐에 등록
                            # 사용자는 기다리지 않게 하고, 생성은 백그라운드(GPU)로 넘김
                            self.enqueue(
                                job_type='music_generation',  # 타입 분리
                                user_uuid=target_job.user_uuid,
                                book_id=target_job.book_id,
                                json_path=full_json_path,
                                music_folder=target_job.music_folder,
                                web_path_prefix=target_job.web_path_prefix
                            )
                            print(f"[Job-CPU] Analysis Done. Music Generation queued for library expansion.")

                        # 사용자에게 보여줄 데이터 준비 완료
                        status = "done"
                    except ImportError:
                        err_msg = "analyzer module missing"
                    except Exception as e:
                        err_msg = str(e)

                # =========================================================
                # [Case B] 음악 생성 작업 (GPU Worker 담당)
                # 목적: 키워드 기반 신규 음악 생성 -> 라이브러리(Storage) 확장
                # =========================================================
                elif target_job.type == "music_generation":
                    print(f"[Job-GPU] Generating New Music for Library: {target_job.book_id}")

                    # (1) 음악 생성 로직 실행
                    # 기존 음악이 매핑되어 있더라도, 다양성을 위해(또는 필요 시) 생성 로직 수행
                    process_book_background(
                        target_job.json_path,
                        target_job.music_folder,
                        target_job.web_path_prefix,
                        target_job.user_uuid,
                        target_job.book_id
                    )

                    # (2) 인덱스 갱신 (생성된 음악을 다음 사용자가 쓸 수 있도록 등록)
                    if create_music_index:
                        try:
                            create_music_index()
                        except:
                            pass

                    status = "done"

                else:
                    status = "skipped"

            except Exception as e:
                err_msg = str(e)
                print(f"[Job] Execution Failed: {e}")
                import traceback
                traceback.print_exc()

            # 4. 결과 저장
            try:
                target_job.status = status
                target_job.finished_at = int(time.time())
                target_job.error = err_msg
                self.db.session.commit()
            except Exception as e:
                self.db.session.rollback()
                print(f"[DB] Failed to save job status: {e}")

        return {"ran": processed_count}