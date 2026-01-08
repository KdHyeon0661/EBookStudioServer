import os
import json
import librosa
import numpy as np
import warnings
import hashlib
import shutil
from filelock import FileLock, Timeout  # pip install filelock 필요

# 경고 무시
warnings.filterwarnings('ignore')

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS_DIR = os.path.join(BASE_DIR, 'defaults')
MUSIC_FOLDER = os.path.join(DEFAULTS_DIR, 'music')
INDEX_FILE = os.path.join(DEFAULTS_DIR, 'music_index.json')
LOCK_FILE = os.path.join(DEFAULTS_DIR, 'music_index.json.lock')


def analyze_audio(file_path):
    """
    오디오 전체 길이 및 BPM 측정 (최적화)
    """
    try:
        # 1. 길이 측정
        try:
            duration = librosa.get_duration(path=file_path)
        except:
            y_temp, sr_temp = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y_temp, sr=sr_temp)

        # 2. BPM 측정 (60초 샘플링)
        y, sr = librosa.load(file_path, sr=None, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        if isinstance(tempo, np.ndarray):
            tempo = tempo[0] if len(tempo) > 0 else 0.0

        return duration, float(tempo)
    except Exception as e:
        print(f"[Error] Analysis failed ({os.path.basename(file_path)}): {e}")
        return 0.0, 0.0


def create_music_index():
    print(f"[Indexer] Start")
    print(f"[Indexer] Scanning target: {MUSIC_FOLDER}")

    if not os.path.exists(MUSIC_FOLDER):
        os.makedirs(MUSIC_FOLDER, exist_ok=True)
        return

    # 락 획득 시도 (최대 60초 대기)
    try:
        with FileLock(LOCK_FILE, timeout=60):
            _process_indexing_critical_section()
    except Timeout:
        print("[Indexer] Failed to acquire lock (Timeout). Another process is running.")
    except Exception as e:
        print(f"[Indexer] Error: {e}")


def _process_indexing_critical_section():
    """
    락이 걸린 상태에서 안전하게 실행되는 실제 인덱싱 로직
    """
    # 2. 디스크 파일 스캔 (하위 폴더 포함)
    valid_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')

    # disk_files_map: { "파일명": "전체경로" }
    disk_files_map = {}

    for root, dirs, files in os.walk(MUSIC_FOLDER):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                disk_files_map[file] = full_path

    print(f"[Indexer] Total audio files found: {len(disk_files_map)}")

    # 3. 기존 인덱스 로드
    index_data = {}
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        except:
            index_data = {}

    changed = False

    # 4. [정리] 디스크에서 삭제된 파일 제거
    ids_to_remove = []
    for key, val in index_data.items():
        if val.get('filename') not in disk_files_map:
            ids_to_remove.append(key)

    if ids_to_remove:
        print(f"[Indexer] Cleaning up {len(ids_to_remove)} removed files...")
        for key in ids_to_remove:
            del index_data[key]
        changed = True

    # 5. [추가] 신규 파일 인덱싱
    registered_filenames = {v['filename'] for v in index_data.values()}
    new_filenames = [f for f in disk_files_map.keys() if f not in registered_filenames]

    if new_filenames:
        print(f"[Indexer] Analyzing {len(new_filenames)} new files...")

    for i, filename in enumerate(new_filenames):
        print(f"   [{i + 1}/{len(new_filenames)}] {filename} ... ", end='', flush=True)

        file_path = disk_files_map[filename]
        json_path = os.path.splitext(file_path)[0] + ".json"

        # 메타데이터 기본값
        genre = "unknown"
        prompt = ""
        bpm = 0

        # 사이드카 JSON 확인
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    meta = json.load(jf)
                    genre = meta.get('genre') or meta.get('emotion') or genre
                    prompt = meta.get('original_prompt') or meta.get('prompt') or prompt
                    bpm = meta.get('bpm', 0)
            except:
                pass

        if genre == "unknown":
            parts = filename.split('_')
            if len(parts) > 1 and parts[1].isalpha():
                genre = parts[1]

        # 오디오 분석
        dur, detected_bpm = analyze_audio(file_path)
        if not bpm: bpm = detected_bpm

        # ID 생성 (해시 충돌 방지)
        file_hash = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 10000000
        while str(file_hash) in index_data:
            file_hash += 1

        index_data[str(file_hash)] = {
            "id": file_hash,
            "genre": genre,
            "filename": filename,
            "duration": round(dur, 2),
            "bpm": int(round(bpm)),
            "prompt": prompt if prompt else f"{genre} mood music"
        }
        changed = True
        print("Done.")

    # 6. 저장 (Atomic Write 적용)
    if changed:
        try:
            # (1) 임시 파일에 먼저 쓰기
            temp_file = INDEX_FILE + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

            # (2) 원자적 교체 (운영체제 레벨에서 안전함)
            os.replace(temp_file, INDEX_FILE)
            print(f"[Indexer] Indexing completed and saved (Total {len(index_data)} items)")
        except Exception as e:
            print(f"[Indexer] Failed to save index: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        print("[Indexer] No changes detected.")


if __name__ == "__main__":
    create_music_index()